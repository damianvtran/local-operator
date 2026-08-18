"""The ``agent`` tool: discover, create, and refine reusable agent profiles.

WHY A TOOL RATHER THAN A FIXED TABLE
====================================

Roles (reviewer, coder, architect, manager, ...) are not harness constants.
Which roles exist, what they are told, and when they apply differ per operator
and per repository, and the guidance only improves if whoever notices it was
wrong can change it — including an agent, mid-session, right after the bad
delegation that revealed the gap.

So the profiles live in the user's own agent registry
(:mod:`local_operator.agents`), the harness ships a handful of starter profiles
as ordinary editable markdown (:mod:`local_operator.agent_profiles`), and this
tool is how an agent works with them:

- ``list`` / ``show`` — what roles exist and what they say.
- ``search`` — which existing role fits a task, by meaning rather than by
  exact name, over the same local embedding index that routes skills.
- ``install`` — pull a packaged starter into the registry on first need, so a
  fresh machine can delegate to a ``reviewer`` without the operator having
  authored one.
- ``create`` / ``update`` — author a new role, or fix one whose instructions
  produced a bad run. ``when_to_use`` is stored as the routing description, so
  a role written today is discoverable by ``search`` tomorrow.

CONTEXT DISCIPLINE
==================

The registry is NEVER enumerated into the prompt: it can be large and its
descriptions are private user content. ``list`` returns one line per profile
and only when called; ``show`` is the only op that returns a full instruction
body. This mirrors how skills are handled — semantic routing decides what is
relevant, and the body loads on demand.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from local_operator.agent_profiles import (
    MAX_INSTRUCTIONS_CHARS,
    AgentProfile,
    NameTakenError,
    install_seed,
    is_role,
    list_seeds,
    load_seed,
    profile_from_agent,
    seed_tags,
)
from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    AgentToolUpdate,
    ToolContext,
    ToolResult,
)
from local_operator.tools.builtin import (
    _error,
    _guard,
    _text,
    _validation_error,
    spill_truncate,
)

logger = logging.getLogger(__name__)


class AgentParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["list", "show", "search", "install", "create", "update"] = Field(
        description=(
            "search: find a role by meaning; list/show: what exists and what it "
            "says; install: add a packaged starter; create/update: author or fix "
            "a role."
        )
    )
    name: str | None = Field(default=None, description="Role name (all ops but search).")
    query: str | None = Field(default=None, description="search: the task, in a sentence.")
    description: str | None = Field(
        default=None,
        description=(
            "create/update: when to use this role. Semantic routing matches this, "
            "so write the trigger condition, not a job title."
        ),
    )
    instructions: str | None = Field(
        default=None,
        description=(
            "create/update: standing guidance prepended to every run of the role. "
            "Imperative and short — it is billed on each of that role's turns."
        ),
    )
    tools: list[str] | None = Field(
        default=None,
        description="create/update: restrict the role to these tools. Omit for all.",
    )
    effort: str | None = Field(
        default=None, description="create/update: default model tier (lo/med/hi)."
    )
    delegate: bool | None = Field(
        default=None,
        description=(
            "create/update: may this role launch its own subagents? Default no — "
            "only coordinating roles should."
        ),
    )


def _profile_line(profile: AgentProfile, *, installed: bool) -> str:
    """One scannable row: name, where it came from, and what it is for."""

    marks: list[str] = [] if installed else ["starter"]
    if profile.tools:
        marks.append(f"{len(profile.tools)} tools")
    if profile.effort:
        marks.append(profile.effort)
    suffix = f" [{', '.join(marks)}]" if marks else ""
    summary = (profile.when_to_use or profile.description or "").strip()
    return f"- {profile.name}{suffix}: {summary}"[:400]


def _registry(context: ToolContext | None) -> Any:
    return getattr(context, "agent_registry", None) if context else None


def _registered_profiles(registry: Any) -> list[AgentProfile]:
    """Registry rows that carry a role, newest listing order aside.

    Only profiles tagged ``role`` are returned. A registry also holds ordinary
    conversational agents and autosave rows, and listing those as delegation
    targets would be noise at best and a privacy leak at worst.
    """

    from local_operator.agent_profiles import profile_from_agent

    try:
        agents = registry.list_agents()
    except Exception:  # noqa: BLE001 - registry problems degrade to "no roles"
        logger.warning("agent registry listing failed")
        return []
    profiles: list[AgentProfile] = []
    for agent in agents:
        # One shared predicate with role RESOLUTION: two readers that disagree
        # about what a role is are how listing came to hide a row that
        # ``task(agent=...)`` would still have run.
        if not is_role(agent):
            continue
        try:
            profiles.append(profile_from_agent(registry, agent))
        except Exception:  # noqa: BLE001
            continue
    return sorted(profiles, key=lambda profile: profile.name.lower())


async def _op_list(context: ToolContext | None, tool_call_id: str) -> ToolResult:
    registry = _registry(context)
    installed = _registered_profiles(registry) if registry is not None else []
    installed_names = {profile.name.lower() for profile in installed}
    lines = [_profile_line(profile, installed=True) for profile in installed]
    starters = [
        profile
        for profile in (load_seed(name) for name in list_seeds())
        if profile is not None and profile.name.lower() not in installed_names
    ]
    body = ""
    if lines:
        body += "registered roles:\n" + "\n".join(lines)
    if starters:
        if body:
            body += "\n\n"
        body += "installable starters (op='install'):\n" + "\n".join(
            _profile_line(profile, installed=False) for profile in starters
        )
    if not body:
        body = "no roles registered and no starters packaged."
    text, spill = spill_truncate(body, "agent", context)
    return _text(tool_call_id, "agent", text, details=spill or None)


async def _op_show(context: ToolContext | None, tool_call_id: str, name: str) -> ToolResult:
    from local_operator.agent_profiles import resolve_profile

    profile = resolve_profile(name, registry=_registry(context))
    if profile is None:
        return _error(tool_call_id, "agent", f"no role named {name!r} (try op='list')")
    origin = "registered" if profile.agent_id else "packaged starter (not installed)"
    header = [
        f"{profile.name} — {origin}",
        f"when to use: {profile.when_to_use or profile.description or '(unstated)'}",
        f"tools: {', '.join(profile.tools) if profile.tools else 'full inventory'}",
    ]
    if profile.effort:
        header.append(f"effort: {profile.effort}")
    body = "\n".join(header) + "\n\ninstructions:\n" + (profile.instructions or "(none)")
    text, spill = spill_truncate(body, "agent", context)
    return _text(tool_call_id, "agent", text, details=spill or None)


async def _op_search(context: ToolContext | None, tool_call_id: str, query: str) -> ToolResult:
    """Rank roles against a task description.

    Uses the offline :class:`LocalEmbedder` over the same index used for
    skills, so a role added a minute ago is searchable without a rebuild step
    and no registry text ever leaves the machine.
    """

    from local_operator.skills.discovery import Skill
    from local_operator.skills.embeddings import LocalEmbedder
    from local_operator.skills.index import SkillIndex

    registry = _registry(context)
    candidates = _registered_profiles(registry) if registry is not None else []
    known = {profile.name.lower() for profile in candidates}
    candidates += [
        profile
        for profile in (load_seed(name) for name in list_seeds())
        if profile is not None and profile.name.lower() not in known
    ]
    if not candidates:
        return _text(tool_call_id, "agent", "no roles available to search.")

    rows = [
        Skill(
            name=profile.name,
            description=(profile.when_to_use or profile.description or profile.name)[:512],
            file_path=Path(f"{profile.name}.md"),
            base_dir=Path("."),
            source="agent-profile",
            resource_type="agent_hint",
        )
        for profile in candidates
    ]
    by_name = {profile.name: profile for profile in candidates}
    try:
        index = SkillIndex(rows, LocalEmbedder())
        await index.build()
        picked = await index.select(query, k=3)
    except Exception as exc:  # noqa: BLE001 - degrade to listing, never fail
        logger.warning("role search failed: %s", exc)
        picked = rows[:3]
    if not picked:
        return _text(
            tool_call_id,
            "agent",
            "no role matched that task closely; use op='list' or launch task without a role.",
        )
    lines = []
    for row in picked:
        profile = by_name.get(row.name)
        if profile is None:
            continue
        lines.append(_profile_line(profile, installed=bool(profile.agent_id)))
    return _text(
        tool_call_id,
        "agent",
        "best matching roles:\n" + "\n".join(lines) + "\n\nlaunch with task(agent='<name>').",
    )


async def _op_install(context: ToolContext | None, tool_call_id: str, name: str) -> ToolResult:
    registry = _registry(context)
    if registry is None:
        return _error(
            tool_call_id, "agent", "no agent registry attached to this session; cannot install."
        )
    if load_seed(name) is None:
        return _error(
            tool_call_id,
            "agent",
            f"no packaged starter named {name!r}; starters: {', '.join(list_seeds())}",
        )
    try:
        profile = install_seed(name, registry=registry)
    except NameTakenError:
        return _error(
            tool_call_id,
            "agent",
            f"an agent named {name!r} already exists and is not a role, so nothing was "
            "installed. Rename that agent, or author the role under a different name "
            "with op='create'.",
        )
    if profile is None:
        return _error(tool_call_id, "agent", f"could not install starter {name!r}")
    return _text(
        tool_call_id,
        "agent",
        f"installed role {profile.name!r} into the registry; "
        f"launch with task(agent={profile.name!r}). Edit it with op='update'.",
    )


async def _op_write(
    context: ToolContext | None,
    tool_call_id: str,
    params: AgentParams,
    *,
    creating: bool,
) -> ToolResult:
    from local_operator.agents import AgentEditFields

    registry = _registry(context)
    if registry is None:
        return _error(
            tool_call_id, "agent", "no agent registry attached to this session; cannot save roles."
        )
    name = (params.name or "").strip()
    try:
        existing = registry.get_agent_by_name(name)
    except Exception:  # noqa: BLE001
        existing = None

    # A name occupied by a NON-role is refused on both paths rather than being
    # converted into one: an ordinary chat agent silently acquiring a role's
    # tags and guidance is a surprising way to lose an agent, and on the update
    # path it would be the same fail-open hijack the role tag exists to stop.
    if existing is not None and not is_role(existing):
        return _error(
            tool_call_id,
            "agent",
            f"an agent named {name!r} exists and is not a role; rename it, or use a "
            "different name for the role.",
        )
    if creating and existing is not None:
        return _error(
            tool_call_id,
            "agent",
            f"role {name!r} already exists; use op='update' to change it.",
        )
    if not creating and existing is None:
        return _error(tool_call_id, "agent", f"no registered role named {name!r} to update.")

    instructions = (params.instructions or "").strip()
    if creating and not instructions:
        return _error(tool_call_id, "agent", "create needs 'instructions' — the role's guidance.")
    if len(instructions) > MAX_INSTRUCTIONS_CHARS:
        return _error(
            tool_call_id,
            "agent",
            f"instructions exceed {MAX_INSTRUCTIONS_CHARS} chars; they ride in front of every "
            "run of this role, so they must stay short.",
        )

    # An UPDATE merges onto the stored profile; only a CREATE starts from
    # nothing. Rebuilding the tags from this call's params alone silently
    # stripped whatever it did not mention — so refining a reviewer's wording
    # (the very operation this tool advertises) dropped its `tools:` allowlist
    # and handed the role the full write inventory, failing OPEN and reporting
    # success. An omitted field means "leave it alone", never "clear it";
    # clearing is done by naming the field with an empty value.
    current = profile_from_agent(registry, existing) if existing is not None else None

    if params.tools is not None:
        tools = tuple(params.tools) or None
    else:
        tools = current.tools if current is not None else None

    if params.effort is not None:
        effort = params.effort.strip() or None
    else:
        effort = current.effort if current is not None else None

    if params.delegate is not None:
        may_delegate = params.delegate
    else:
        may_delegate = current.may_delegate if current is not None else False

    description = (params.description or "").strip()
    if not description and current is not None:
        description = current.description

    profile = AgentProfile(
        name=name,
        description=description,
        when_to_use=description,
        instructions=instructions,
        tools=tools,
        effort=effort,
        # Settable, and preserved when not set, for the same reason as the
        # allowlist: a role that coordinates must not stop coordinating
        # because someone fixed a typo in its guidance — but a role authored
        # here must also be ABLE to coordinate, which previously required
        # hand-editing the tags.
        may_delegate=may_delegate,
    )
    tags = list(seed_tags(profile))

    # Every field is spelled out (``AgentEditFields`` is validated in strict
    # mode, and every other caller in the tree does the same): a role inherits
    # the session's model and sampling settings rather than pinning any.
    def _fields(**overrides: Any) -> AgentEditFields:
        base: dict[str, Any] = dict(
            name=None,
            description=None,
            tags=None,
            categories=None,
            security_prompt=None,
            hosting=None,
            model=None,
            last_message=None,
            temperature=None,
            top_p=None,
            top_k=None,
            max_tokens=None,
            stop=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            current_working_directory=None,
        )
        base.update(overrides)
        return AgentEditFields(**base)

    if existing is None:
        agent = registry.create_agent(
            _fields(name=name, description=profile.description, tags=tags, categories=["role"])
        )
    else:
        agent = existing
        # An update carries only what changed: passing a None description here
        # would blank the routing text a previous create had set.
        overrides: dict[str, Any] = {"tags": tags}
        if profile.description:
            overrides["description"] = profile.description
        registry.update_agent(agent.id, _fields(**overrides))
    if instructions:
        registry.set_agent_system_prompt(agent.id, instructions)
    verb = "created" if existing is None else "updated"
    return _text(
        tool_call_id,
        "agent",
        f"{verb} role {name!r}; launch with task(agent={name!r}).",
    )


@_guard("agent")
async def execute_agent(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Discover, install, and author reusable agent role profiles.

    ``@_guard`` for the same reason every sibling executor has it: the harness
    contract is that tools never throw into the loop, and this one has live
    raising paths — a read-only agents directory (PermissionError from
    create/install) and a transient registry read failure followed by a name
    that does exist (ValueError out of ``create_agent``'s uniqueness check).
    The loop's outer net would catch those, but without the traceback tail that
    makes them debuggable from a transcript.
    """

    try:
        params = AgentParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "agent", exc)

    if params.op in {"show", "install", "create", "update"} and not (params.name or "").strip():
        return _error(tool_call_id, "agent", f"op={params.op!r} needs 'name'.")
    if params.op == "search" and not (params.query or "").strip():
        return _error(tool_call_id, "agent", "op='search' needs 'query'.")

    if params.op == "list":
        return await _op_list(context, tool_call_id)
    if params.op == "show":
        return await _op_show(context, tool_call_id, str(params.name))
    if params.op == "search":
        return await _op_search(context, tool_call_id, str(params.query))
    if params.op == "install":
        return await _op_install(context, tool_call_id, str(params.name))
    return await _op_write(context, tool_call_id, params, creating=params.op == "create")


def build_agent_tool(context: ToolContext) -> AgentTool | None:
    """createIf: the tool exists only where a registry can back it.

    Without a registry the read-only ops would still work off packaged
    starters, but ``install``/``create``/``update`` could not persist anything
    — a tool that can show a role and never keep one is a worse surface than
    no tool, so it is not advertised at all.
    """

    if getattr(context, "agent_registry", None) is None:
        return None
    return AgentTool(
        name="agent",
        label="Agent roles",
        description=(
            "Reusable role profiles for delegation (reviewer, coder, architect, "
            "manager, designer, scout). Find, install, or author one; launch it "
            "with task(agent='<name>')."
        ),
        parameters=AgentParams.model_json_schema(),
        # Writes land in the user's own configuration directory, never in the
        # workspace, and are trivially reversible by editing the profile back.
        # Gating them behind an approval prompt would make an agent improving
        # its own reviewer guidance an interruption, which is exactly the
        # friction that keeps the registry empty.
        approval_tier="read",
        concurrency="exclusive",
        interruptible=False,
        execute=execute_agent,
    )
