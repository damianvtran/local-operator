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
    DESCRIPTION_MAX,
    HISTORY_DEFAULT_TAIL,
    MILESTONES_MAX,
    PROGRESS_MAX,
    SESSIONS_MAX,
    UPDATES_MAX,
    MilestoneEdit,
    Project,
    ProjectEdit,
    ProjectMilestone,
    ProjectNameConflictError,
    ProjectRegistry,
    ProjectRegistryLockTimeout,
    ProjectSchemaGuardError,
    ProjectStatus,
    build_project_view,
    display_name,
    history_lines,
    milestone_status,
    progress_date_text,
    progress_is_stale,
    readable_error,
    refreshed_age_text,
    refreshed_note,
    reported_age,
    stale_after_s,
    truncate_row,
)
from local_operator.tools.builtin import (
    _error,
    _guard,
    _text,
    _validation_error,
    spill_truncate,
)

logger = logging.getLogger(__name__)


class ProjectParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["list", "show", "create", "update", "refresh", "link", "unlink", "milestone"] = (
        Field(description="The verb to run.")
    )
    name: str | None = Field(default=None, description="Project name (all ops but list).")
    description: str | None = Field(
        default=None,
        # Resolution note (fold onto #1837): upstream's text WINS here. Issue
        # #1815 deliberately ties the schema text, the guide and the refusal
        # together (test_the_schema_text_names_the_cap_the_refusal_enforces),
        # and its contract is the generous markdown promise built from
        # DESCRIPTION_MAX — so this field keeps "multiple paragraphs, headings,
        # lists and code" and the dynamic cap, and the slimming wave's pin for
        # this tool rests on the milestone-rationale phrase instead (the
        # MOVED_WIRE_DETAIL entry in tests/unit/tools/test_tool_docs.py).
        description=(
            "create/update: markdown prose describing the workstream — multiple "
            "paragraphs, headings, lists and code; rendered, not dumped "
            f"(<= {DESCRIPTION_MAX} chars)."
        ),
    )
    owner: str | None = Field(default=None, description="create/update: owner.")
    team: str | None = Field(default=None, description="create/update: managing team.")
    title: str | None = Field(
        default=None,
        description="create/update: display name (`name` stays the key).",
    )
    status: ProjectStatus | None = Field(
        default=None,
        description="create/update: 'done' needs milestones complete or force_done.",
    )
    progress: str | None = Field(
        default=None,
        description="update: one dated line, not a transcript; nothing moved: op='refresh'.",
    )
    tags: list[str] | None = Field(
        default=None,
        description="create/update: replace-set; each [a-z0-9][a-z0-9_-]{0,23}.",
    )
    session_id: str | None = Field(
        default=None, description="link/unlink: defaults to THIS session."
    )
    start_date: str | None = Field(default=None, description="create/update: YYYY-MM-DD.")
    target_date: str | None = Field(
        default=None,
        description="create/update: YYYY-MM-DD, not before start_date.",
    )
    completed_at: str | None = Field(
        default=None,
        description="create/update: YYYY-MM-DD the work finished.",
    )
    estimate: float | None = Field(default=None, description="create/update: > 0, <= 1000.")
    estimate_unit: Literal["points", "days"] | None = Field(
        default=None, description="create/update."
    )
    milestones: list[ProjectMilestone] | None = Field(
        default=None,
        description="create; update needs replace_milestones. One: op='milestone'.",
    )
    replace_milestones: bool = Field(
        default=False,
        description="update: true to replace the WHOLE milestones list.",
    )
    force_done: bool = Field(
        default=False,
        description="create/update: allow 'done' with open milestones.",
    )
    milestone: str | None = Field(default=None, description="milestone: name (add-or-update).")
    milestone_target_date: str | None = Field(
        default=None,
        description="milestone: YYYY-MM-DD.",
    )
    milestone_completed: bool | None = Field(
        default=None,
        description="milestone: true = done today, false clears.",
    )
    remove: bool = Field(default=False, description="milestone: remove the named milestone.")
    attach: list[str] | None = Field(
        default=None,
        description="update: evidence file paths for the new progress line.",
    )
    history: int | None = Field(
        default=None,
        ge=0,
        le=UPDATES_MAX,
        description=(
            "show: how many recent history entries to print (default "
            f"{HISTORY_DEFAULT_TAIL}, max {UPDATES_MAX}; 0 omits the section)."
        ),
    )


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


def _milestone_counts(project: Project) -> str:
    done = sum(1 for m in project.milestones if m.completed_at)
    return f"M {done}/{len(project.milestones)}"


def _estimate_text(project: Project) -> str:
    if project.estimate is None:
        return "no estimate"
    suffix = "pt" if project.estimate_unit == "points" else "d"
    number = f"{project.estimate:g}"
    return f"est {number}{suffix}"


def _row(project: Project, *, now: float | None = None, window: float | None = None) -> str:
    """One scannable listing line, ``PROJECT_ROW_CAP``-bounded (in CELLS)."""
    identity = display_name(project) or "- unnamed"
    if project.title:
        # Title-first with the key as secondary: the key is the addressing
        # handle, so a titled row must keep it recoverable from the listing.
        identity = f"{identity} ({project.name})"
    parts = [
        f"- {identity} [{project.status}]",
        _estimate_text(project),
    ]
    if project.owner:
        parts.append(f"owner: {project.owner}")
    if project.team:
        parts.append(f"team: {project.team}")
    if project.target_date:
        parts.append(f"→{project.target_date}")
    if project.milestones:
        parts.append(_milestone_counts(project))
    # WORKING sessions: ``sessions`` IS the work set, so the count and the
    # completion check can never disagree about whether anyone is on the job;
    # a filing is named separately and never reads as participation.
    sessions = len(project.sessions)
    parts.append(f"{sessions} working session" + ("" if sessions == 1 else "s"))
    filed = len(project.coordination_sessions)
    if filed:
        parts.append(f"{filed} filed")
    age = reported_age(project, now=now)
    if age is None:
        parts.append("no progress")
    else:
        stale = " (stale)" if progress_is_stale(project, now=now, window=window) else ""
        parts.append(f"progress {age} ago{stale}")
        refreshed = refreshed_age_text(project, now=now)
        if refreshed is not None:
            # The refreshed TOKEN beside the age: the badge keeps telling the
            # content clock's truth, and this says the record was checked.
            parts.append(f"refreshed {refreshed} ago")
    summary = (project.description or "").strip()
    row = " · ".join(parts)
    if summary:
        # QUOTED: an unquoted tail read as if it were the progress text (the
        # first cut of this row shipped exactly that ambiguity).
        row += f' · "{summary}"'
    return truncate_row(row)


def _field_lines(
    project: Project, *, history_tail: int = HISTORY_DEFAULT_TAIL, window: float | None = None
) -> list[str]:
    """The ``show`` header block: every stored field, derived status where one exists."""
    lines = [
        f"{display_name(project) or '(unnamed)'} [{project.status}]",
    ]
    if project.title:
        # The display title is the first line; the addressing key must stay
        # visible (it is what op='update'/'link' take).
        lines.append(f"key: {project.name}")
    lines.extend(
        [
            f"description: {project.description or '(unstated)'}",
            f"owner: {project.owner or '(unstated)'}",
            f"team: {project.team or '(unstated)'}",
            f"estimate: {_estimate_text(project)}",
            f"dates: start {project.start_date or '—'} · target {project.target_date or '—'}"
            f" · completed {project.completed_at or '—'}",
            f"tags: {', '.join(project.tags) if project.tags else '(none)'}",
        ]
    )
    age = reported_age(project)
    stale = ", stale" if progress_is_stale(project, window=window) else ""
    freshness = f"reported {age} ago{stale}" if age is not None else "none recorded"
    lines.append(f"progress ({freshness}): {project.progress or '—'}")
    if project.progress and project.progress_reported_by:
        lines.append(f"progress reported by: {project.progress_reported_by}")
    note = refreshed_note(project)
    if note is not None:
        # THE sentence, one copy (``projects.refreshed_note``): "refreshed 1h
        # ago by session X — no new content since 2026-09-29".
        lines.append(note)
    lines.extend(history_lines(project, tail=history_tail))
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
    """Per-session rollup from the ONE shared composition (``build_project_view``).

    WORKING sessions and filings are separate sections: the working list is
    what feeds every liveness reading, and a filing (``role='coordination'``)
    is provenance — it is never rendered with a runtime state (there is none on
    the row) so it can never be misread as participation.
    """
    rows = view.get("sessions") or []
    if not rows:
        return ["linked sessions: (none)"]
    working = [row for row in rows if row.get("role") != "coordination"]
    filed = [row for row in rows if row.get("role") == "coordination"]
    lines: list[str] = []
    if working:
        lines.append(f"working sessions ({len(working)}/{SESSIONS_MAX}):")
        for row in working:
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
    if filed:
        lines.append(f"filed by ({len(filed)}):")
        for row in filed:
            session_id = row["session_id"]
            if not row.get("exists"):
                lines.append(f"  - {session_id} [filed] — no session directory")
                continue
            title = row.get("title") or "(untitled)"
            bits = ["filed"]
            if row.get("archived"):
                bits.append("archived")
            lines.append(f"  - {session_id} [{' · '.join(bits)}] {title}")
    return lines


def _project_edit(params: ProjectParams, *, creating: bool) -> ProjectEdit:
    """A ``ProjectEdit`` carrying exactly the fields the caller supplied.

    ``model_fields_set`` is the whole point: an update must not clobber a field
    the caller never mentioned, and an explicit ``""``/``None`` must stay
    distinguishable from an absent one (the clear-vs-untouched vocabulary).

    An over-cap TEXT field is refused HERE, before the store's own
    ``max_length`` backstop can answer with the bare pydantic sentence
    ("String should have at most 2000 characters") — the refusal that cost a
    reporter two blind retries (issue #1815). Each one names the field, the
    submitted size, the exact limit and the remedy, in the same
    ``(submitted n)`` shape the sibling tools' refusals use.

    BOTH prose fields are checked (`progress` as well as `description`): the
    description's remedy points the writer at progress lines, so a long
    progress line is the FIRST thing that remedy reaches — leaving it to the
    store's backstop handed back the same bare sentence this issue is about
    (design review round 1, D2).
    """
    description = params.description
    if description is not None and len(description) > DESCRIPTION_MAX:
        raise ValueError(
            f"project 'description' is over the {DESCRIPTION_MAX}-character cap "
            f"(submitted {len(description)}) — shorten it, or keep the long detail "
            "in progress lines (op='update'), which the history keeps."
        )
    progress = params.progress
    if progress is not None and len(progress) > PROGRESS_MAX:
        raise ValueError(
            f"project 'progress' is over the {PROGRESS_MAX}-character cap "
            f"(submitted {len(progress)}) — shorten it; earlier updates stay in the history."
        )
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
        # The REPLACE guard's switch: consumed by `_op_update`, never a row
        # field (ProjectEdit forbids extras, so a leak here would be a
        # validation error on every deliberate replace).
        "replace_milestones",
        # The DONE gate's switch: consumed by `_op_create`/`_op_update`, never
        # a row field — same reason.
        "force_done",
        "attach",
        "history",
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
            "session automatically) once the workstream is worth tracking — with "
            "a short `title` and a markdown `description`. Read "
            "guide://projects first."
        )
        return _text(tool_call_id, "project", body)
    body = "projects:\n" + "\n".join(
        _row(project, window=stale_after_s(registry.config_dir)) for project in projects
    )
    body += "\n\nop='show' name='<name>' has the full record plus its sessions."
    text, spill = spill_truncate(body, "project", context)
    return _text(tool_call_id, "project", text, details=spill or None)


async def _op_show(
    context: ToolContext | None, tool_call_id: str, name: str, *, history_tail: int
) -> ToolResult:
    registry = _registry(context)
    if registry is None:
        return _error(tool_call_id, "project", "no project registry attached to this session.")
    try:
        project = registry.get_project_by_name(name)
    except ProjectRegistryLockTimeout as exc:
        return _error(tool_call_id, "project", str(exc))
    if project is None:
        return _error(tool_call_id, "project", f"no project named {name!r} (try op='list')")
    lines = _field_lines(
        project,
        history_tail=history_tail,
        window=stale_after_s(registry.config_dir),
    )
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
    # The ROLE decision belongs to the write surface: the chief of staff's
    # create-time auto-link is provenance ("filed by"), never participation —
    # her session is live almost always, and a working link would make every
    # row she files read as "someone is on it" and earn her a nudge for work
    # she does not do. Every other caller keeps the working auto-link.
    coordination = False
    if session_id:
        from local_operator.aida.state import is_aida_session

        coordination = is_aida_session(registry.config_dir, session_id)
    try:
        fields = _project_edit(params, creating=True)
        project = registry.create_project(
            fields,
            sessions=[] if coordination else ([session_id] if session_id else []),
            coordination_sessions=[session_id] if (session_id and coordination) else [],
            force_done=params.force_done,
        )
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
    forced = ""
    if (
        params.force_done
        and project.status == "done"
        and any(milestone.completed_at is None for milestone in project.milestones)
    ):
        # The update path's deliberate-act rule applies to the other path too
        # (agent review round 1, N2): a creation that closed over open
        # milestones must say so, or the receipt reads as an ordinary create.
        forced = "status 'done' forced with milestones incomplete (force_done=true). "
    if session_id and coordination:
        receipt = (
            f"created project {project.name!r} [{project.status}]; filed by this session "
            f"(session {session_id}) — a coordination link, not a working session. "
        )
    elif session_id:
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
        + forced
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
    if (
        "milestones" in params.model_fields_set
        and params.milestones is not None
        and not params.replace_milestones
    ):
        # The incident this guard exists for: an agent read "update a milestone"
        # as `op='update'` with a one-entry list, and the store did what its
        # field said — replaced the whole list, silently wiping the siblings.
        # Refuse BEFORE any edit is built; the safe path is one op away. The
        # `is not None` arm keeps an explicit `milestones=null` the no-op the
        # store has always made of it — refusing there would drop the call's
        # other fields, which applied pre-guard (agent review round 1, M2).
        return _error(
            tool_call_id,
            "project",
            f"update would REPLACE all milestones ({len(project.milestones)} currently "
            "stored). Use op='milestone' to add/update/remove ONE milestone by name, "
            "or pass replace_milestones=true to replace the list deliberately.",
        )
    reporter = _calling_session_id(context) or "operator"
    try:
        fields = _project_edit(params, creating=False)
        outcome = registry.update_project(
            project.id,
            fields,
            reporter=reporter,
            attachments=params.attach or (),
            force_done=params.force_done,
        )
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
    # The receipt must NAME a deliberate replace wherever the call made one — a
    # bare "updated project" is exactly the sentence that hid the wipe this
    # guard stops, and the refreshed branch below used to return before the
    # note was appended (agent review round 1, M1). A supplied LIST under the
    # flag is what constitutes a replace; an explicit `milestones=null`
    # replaces nothing and must not be claimed (M2).
    replaced = ""
    if (
        params.replace_milestones
        and "milestones" in params.model_fields_set
        and params.milestones is not None
    ):
        replaced = (
            f"; milestones replaced deliberately ({_milestone_counts(updated)}; "
            "replace_milestones=true)"
        )
    forced = ""
    if (
        params.force_done
        and updated.status == "done"
        and any(milestone.completed_at is None for milestone in updated.milestones)
    ):
        # A forced close is as deliberate as a replaced list, and this receipt
        # names deliberate acts (the M1 rule): without the clause, a plan that
        # closed over open milestones would read as an ordinary update.
        forced = "; status 'done' forced with milestones incomplete (force_done=true)"
    if outcome.refreshed:
        # Refresh ≠ update: the receipt must not claim the record became
        # fresher — the CHECK is dated, the line is not. The sentence names
        # the content's own date so the model never reads it as "re-dated".
        day = progress_date_text(updated.progress_updated_at)
        still = (
            f"the line still dates from {day}"
            if day is not None
            else "the line keeps its original date"
        )
        return _text(
            tool_call_id,
            "project",
            f"refreshed project {updated.name!r} — progress unchanged; {still} "
            f"(no new content){replaced}{forced}.",
        )
    age = reported_age(updated)
    detail = f"progress {age} ago" if age is not None else "no progress recorded"
    if "progress" not in params.model_fields_set:
        # Only mention the snippet when the call itself touched it: a
        # status-only update that says "progress 3h ago" invites the model to
        # think it reported something.
        detail = f"status {updated.status}"
    detail += replaced
    detail += forced
    stored = len(params.attach or ())
    if stored:
        detail += f", {stored} attachment{'s' if stored != 1 else ''} stored"
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


async def _op_refresh(
    context: ToolContext | None, tool_call_id: str, params: ProjectParams
) -> ToolResult:
    """The textless refresh: record that the stored line still describes reality.

    Nothing is written when the record is fresh ("no reason to send it every
    turn") or carries no progress at all (there is nothing to assert about —
    the first honest line is the right act). A stale record gets the assertion
    pair and NEVER moves the content clock: the stale badge keeps reading the
    truth, while the assertion quiets the completion check for one window.
    """
    if "progress" in params.model_fields_set:
        return _error(
            tool_call_id,
            "project",
            "op='refresh' is textless: drop 'progress' — a new line is op='update' "
            "progress=<text>, and re-sending the identical line already records a "
            "refresh.",
        )
    registry = _registry(context)
    if registry is None:
        return _error(tool_call_id, "project", "no project registry attached to this session.")
    name = (params.name or "").strip()
    try:
        project = registry.get_project_by_name(name)
    except ProjectRegistryLockTimeout as exc:
        return _error(tool_call_id, "project", str(exc))
    if project is None:
        return _error(tool_call_id, "project", f"no project named {name!r} (try op='list').")
    if not (project.progress or "").strip():
        return _error(
            tool_call_id,
            "project",
            f"project {project.name!r} has no recorded progress to refresh — write the "
            "first line with op='update' progress=<text>.",
        )
    reporter = _calling_session_id(context) or "operator"
    try:
        outcome = registry.refresh_project(project.id, reporter=reporter)
    except ProjectRegistryLockTimeout as exc:
        return _error(tool_call_id, "project", str(exc))
    except ProjectSchemaGuardError as exc:
        return _error(tool_call_id, "project", str(exc))
    except (ValueError, ValidationError) as exc:
        return _error(tool_call_id, "project", readable_error(exc))
    except Exception as exc:  # noqa: BLE001
        return _error(tool_call_id, "project", f"could not refresh the project: {exc}")
    if not outcome.changed:
        return _text(
            tool_call_id,
            "project",
            f"project {outcome.project.name!r} is not stale — nothing written "
            "(a refresh is for a stale record; there is no reason to send it every turn).",
        )
    refreshed = outcome.project
    day = progress_date_text(refreshed.progress_updated_at)
    still = (
        f"the line still dates from {day}"
        if day is not None
        else "the line keeps its original date"
    )
    return _text(
        tool_call_id,
        "project",
        f"refreshed project {refreshed.name!r} — progress unchanged; {still} "
        "(no new content; the stale badge stays until a new line lands).",
    )


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
    # The pre-state decides which receipt is true: linking an id that was
    # FILED moves it between lists, and a receipt that said "linked" would
    # hide the one act a reader must know about (its liveness role changed).
    was_filed = session_id in project.coordination_sessions
    # SELF-FILING (P1's write policy at the link surface): when the CHIEF OF
    # STAFF links HERSELF, the link is provenance, not work — the create-time
    # auto-link's role decision, applied here so a self-link can never quietly
    # flip her from filed to working. Two edges, both deliberate:
    #  - `session_id not in project.sessions` keeps an EXISTING work link
    #    untouched (a self-link on a row she genuinely works on is a no-op,
    #    never a demotion);
    #  - the check reads TARGET == CALLER, so an explicit `link` naming her id
    #    from ANOTHER session still lands as work — the "a wrong demotion is
    #    one op='link' from restored" repair path (ruling §2.3).
    self_filing = False
    if (
        linking
        and session_id == _calling_session_id(context)
        and session_id not in project.sessions
    ):
        from local_operator.aida.state import is_aida_session

        self_filing = is_aida_session(registry.config_dir, session_id)
    try:
        if linking:
            project, changed = registry.link_session(
                project.id, session_id, role="coordination" if self_filing else "work"
            )
        else:
            project, changed = registry.unlink_session(project.id, session_id)
    except ProjectRegistryLockTimeout as exc:
        return _error(tool_call_id, "project", str(exc))
    except (ValueError, ValidationError) as exc:
        return _error(tool_call_id, "project", readable_error(exc))
    except Exception as exc:  # noqa: BLE001
        return _error(tool_call_id, "project", f"could not update the link set: {exc}")
    working = f"{len(project.sessions)} working"
    if project.coordination_sessions:
        working += f" + {len(project.coordination_sessions)} filed"
    if linking and self_filing:
        if changed:
            return _text(
                tool_call_id,
                "project",
                f"filed session {session_id} on {project.name!r} — a coordination link, "
                "not a working session.",
            )
        return _text(
            tool_call_id,
            "project",
            f"session {session_id} is already filed on {project.name!r} — a coordination "
            "link, not a working session.",
        )
    if linking and not changed:
        return _text(
            tool_call_id,
            "project",
            f"session {session_id} was already linked to {project.name!r} as a working "
            f"session ({working}).",
        )
    if not linking and not changed:
        return _error(
            tool_call_id,
            "project",
            f"session {session_id} is not linked to {project.name!r}; "
            "op='show' lists its sessions and filings.",
        )
    if linking and was_filed:
        return _text(
            tool_call_id,
            "project",
            f"moved session {session_id} from filed to working links on {project.name!r} "
            f"({working} now).",
        )
    return _text(
        tool_call_id,
        "project",
        f"{'linked' if linking else 'unlinked'} session {session_id} "
        f"{'to' if linking else 'from'} {project.name!r} ({working} now).",
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

    needs_name = {"show", "create", "update", "refresh", "link", "unlink", "milestone"}
    if params.op in needs_name and not (params.name or "").strip():
        return _error(tool_call_id, "project", f"op={params.op!r} needs 'name'.")
    if params.attach and params.op != "update":
        return _error(
            tool_call_id,
            "project",
            "attach works only with op='update': the files ride the history entry a "
            "NEW progress line appends. Create the project first if it does not exist.",
        )
    if params.op == "list":
        return await _op_list(context, tool_call_id)
    if params.op == "show":
        tail = HISTORY_DEFAULT_TAIL if params.history is None else params.history
        return await _op_show(context, tool_call_id, str(params.name), history_tail=tail)
    if params.op == "create":
        return await _op_create(context, tool_call_id, params)
    if params.op == "update":
        return await _op_update(context, tool_call_id, params)
    if params.op == "refresh":
        return await _op_refresh(context, tool_call_id, params)
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
            "Track multi-session workstreams: create a project (auto-linked; "
            "filed as coordination when the caller is the chief of staff), "
            "report honest progress (op='refresh' when you checked and nothing "
            "moved), link sessions, set dates/estimate/milestones, read the "
            "aggregated view. Read guide://projects first."
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
