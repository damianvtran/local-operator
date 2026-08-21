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
            "a role or a specialist profile."
        )
    )
    name: str | None = Field(
        default=None, description="Role or specialist name (all ops but search)."
    )
    query: str | None = Field(default=None, description="search: the task, in a sentence.")
    description: str | None = Field(
        default=None,
        description=(
            "create/update: when to use this profile. Semantic routing matches "
            "this, so write the trigger condition, not a job title."
        ),
    )
    instructions: str | None = Field(
        default=None,
        description=(
            "create/update: standing guidance prepended to every run of the "
            "profile. Imperative and short — it is billed on each of that "
            "profile's turns. This is the BASE behaviour; a team layers "
            "collaboration and project briefs on top without rewriting it."
        ),
    )
    tools: list[str] | None = Field(
        default=None,
        description="create/update: restrict the profile to these tools. Omit for all.",
    )
    effort: Literal["lo", "med", "hi", ""] | None = Field(
        default=None,
        description="create/update: default model tier. '' clears it.",
    )
    delegate: bool | None = Field(
        default=None,
        description=(
            "create/update: may this profile launch its own subagents? Default "
            "no — only coordinating roles should."
        ),
    )
    kind: Literal["role", "specialist"] | None = Field(
        default=None,
        description=(
            "create: 'role' (default) is a reusable delegation target tagged "
            "for task(agent=...). 'specialist' is a durable named agent with "
            "its own instruction set — a User Dashboard Agent, a support "
            "triager — that can sit on a team roster without being a role. "
            "Ignored on update: a profile cannot change kind."
        ),
    )


#: Cap on one rendered row. Two lines at 80 columns: enough for a role to say
#: what it is for, short enough that six of them stay scannable in a 24-row
#: terminal. It was 400, which let the enriched retrieval text render six roles
#: as 22 physical lines with no indent to mark where one ended.
_ROW_CAP = 160

#: Hybrid score at or above which a search result is worded as a match rather
#: than as a nearest neighbour. Not a cut: everything is still returned, ranked.
#: Measured true-positive floor 0.118, false-positive ceiling ~0.106.
_STRONG_MATCH = 0.115


def _name_taken_message(name: str, *, installing: bool = False) -> str:
    """One wording for the same condition, wherever it is detected.

    `install` and `create`/`update` used to phrase this differently, and only
    one of them stated the no-op guarantee and named the escape route.
    """

    lead = (
        f"an agent named {name!r} already exists and is not a role, so nothing was installed."
        if installing
        else f"an agent named {name!r} already exists and is not a role."
    )
    return f"{lead} Rename that agent, or use a different name for the role."


def _profile_line(profile: AgentProfile, *, installed: bool) -> str:
    """One scannable row: name, where it came from, and what it is for."""

    marks: list[str] = [] if installed else ["starter"]
    # ALWAYS say something about the tool surface. Emitting `[7 tools]` only
    # when an allowlist exists inverted the signal it is supposed to carry:
    # `None` means the FULL inventory, so the unrestricted roles (coder,
    # designer — which can edit, write and run anything) rendered as a bare
    # `[starter]` while read-only `scout` showed `[5 tools]` and looked
    # heavier. Tool restriction is the one attribute the guide calls "a
    # capability boundary, not advice", so the row must not communicate its
    # absence by saying nothing. `show` already spells this out; this matches.
    if not profile.tools:
        marks.append("all tools")
    else:
        count = len(profile.tools)
        marks.append(f"{count} tool" if count == 1 else f"{count} tools")
    if profile.effort:
        marks.append(profile.effort)
    suffix = f" [{', '.join(marks)}]" if marks else ""
    # DISPLAY prefers the short ``description``; RETRIEVAL (in ``_op_search``)
    # prefers the long ``when_to_use``. The two texts exist for different
    # readers, and drawing both from the same field is what made the list 69%
    # taller at 80 columns once the trigger text was enriched for search. An
    # INSTALLED role collapses to one registry column (the routing text, so
    # search keeps working across install), which is why the row cap does the
    # remaining work rather than this preference alone. ``show`` is where a
    # role's full applicability is read.
    summary = (profile.description or profile.when_to_use or "").strip()
    if not summary:
        # A role with no description is invisible to `search`, which matches on
        # exactly this text. Saying so is more useful than a dangling colon.
        summary = "(no description — not searchable; add one with op='update')"
    row = f"- {profile.name}{suffix}: {summary}"
    # Ellipsis when the cut fires: a bare slice ends mid-word, and the reader
    # cannot tell an author's fragment from text we dropped.
    return row if len(row) <= _ROW_CAP else row[: _ROW_CAP - 1].rstrip() + "…"


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
        body += (
            "installable starters (packaged, not yet in your registry — op='install'):\n"
        ) + "\n".join(_profile_line(profile, installed=False) for profile in starters)
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
    # The highest-intent moment in the flow: someone has just read a role's
    # guidance and decided they want it. Every other op ends by naming the next
    # command; this one used to stop at "not installed" and leave them there.
    body += (
        f"\n\nlaunch with task(agent={profile.name!r})"
        + ("" if profile.agent_id else f", or op='install' name={profile.name!r} to edit it")
        + "."
    )
    text, spill = spill_truncate(body, "agent", context)
    return _text(tool_call_id, "agent", text, details=spill or None)


async def _ranked_names(index: Any, query: str) -> tuple[list[str], float]:
    """Role names ordered by SCORE (best first), and the best score.

    ``SkillIndex.select`` returns its picks sorted by NAME so that two turns
    selecting the same set render byte-identically for the prompt cache. That
    is right for the knowledge block and wrong here: printing an alphabetical
    list under the words "best first" would be a claim the reader cannot check.
    This recomputes the ranking from the same hybrid score the selection used.

    The score comes back too so the caller can HEDGE its wording: with no
    absolute threshold, an unrelated query still returns three roles, and
    nothing in a bare list distinguishes a 0.65 hit from 0.03 noise.

    Best-effort: any failure returns an empty list and the caller keeps
    ``select``'s own order AND drops its "best first" claim, because an
    unranked answer beats no answer but a false claim about it does not.

    KNOWN COUPLING (accepted, not overlooked): this reaches two private names,
    ``skills.index._hybrid_scores`` and ``index._scores``, to recompute a
    ranking ``_select_with_backend`` already computes and then discards by
    sorting on name. It also passes ``cwd=None``, opting out of the glob boosts
    ``select`` applies — inert today because agent-profile rows carry no globs,
    but it is the seam where the two rankings could diverge. The clean fix is
    for ``select`` to grow an ordering option (or return scores) so the ranking
    has ONE owner; that is a change to shared knowledge-routing code and does
    not belong in this PR. The fallback above is what bounds the blast radius
    until then.
    """

    try:
        from local_operator.skills.index import _hybrid_scores

        query_vector = (await index.backend.embed([query]))[0]
        scores = _hybrid_scores(query, index.skills, index._scores(query_vector), None)
    except Exception:  # noqa: BLE001 - ordering is a nicety, never a failure
        # LOGGED, unlike a bare degrade: the caller cannot tell a working
        # ranking from a broken one by looking at the output, so silence here
        # would make a signature drift in the private helpers below invisible
        # in both the result and the logs.
        logger.warning("role ranking unavailable; falling back to unordered", exc_info=True)
        return [], 0.0
    ranked = sorted(range(len(index.skills)), key=lambda position: -scores[position])
    best = float(scores[ranked[0]]) if ranked else 0.0
    return [index.skills[position].name for position in ranked], best


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
        # threshold=0 and RANKED, deliberately, on both counts.
        #
        # The shared index default (0.19) is calibrated for full skill bodies;
        # a role description is one sentence, so correct matches score
        # 0.118-0.177 and were being cut. Measured against a query set, the
        # true best role ranked first 10/10 while unrelated queries ("order me
        # a pizza") topped out at 0.106 — so the RANKING is trustworthy while
        # the absolute scores sit either side of a knife-edge cut. Showing a
        # ranked shortlist is therefore both more useful and more honest than
        # "no role matched": the caller is an agent choosing among six options,
        # and the copy says these are the closest rather than that they fit.
        #
        # `select` sorts its result by NAME (for prompt-cache stability), which
        # would silently present an alphabetical list as if it were ranked, so
        # the ordering is recovered here from the scores themselves.
        picked = await index.select(query, k=3, threshold=0.0)
        ranking, best_score = await _ranked_names(index, query)
        order = {name: position for position, name in enumerate(ranking)}
        picked = sorted(picked, key=lambda row: order.get(row.name, len(order)))
    except Exception as exc:  # noqa: BLE001 - degrade to listing, never fail
        logger.warning("role search failed: %s", exc)
        picked = rows[:3]
        best_score = 0.0
        ranking = []
    if not picked:
        return _text(
            tool_call_id,
            "agent",
            "no roles available to search; use op='list'.",
        )
    lines = []
    for row in picked:
        profile = by_name.get(row.name)
        if profile is None:
            continue
        lines.append(_profile_line(profile, installed=bool(profile.agent_id)))
    # Measured on a query set: true matches score 0.118 and up, while
    # unrelated queries ("order me a pizza") top out near 0.10. The margin is
    # too thin to CUT on (that was the old behaviour, and it hid roles whose
    # own text promised the query), but wide enough to change the wording, so a
    # weak result reads as a nearest neighbour rather than a recommendation.
    # "(best first)" is claimed ONLY when the ranking actually succeeded. With
    # `ranking` empty the rows keep `select`'s NAME order, so the phrase would
    # be a claim the reader cannot check — the exact hazard `_ranked_names`'
    # docstring names, reintroduced by its own fallback.
    if not ranking:
        header = "closest roles:"
    elif best_score >= _STRONG_MATCH:
        header = "closest roles (best first):"
    else:
        header = "nothing scored strongly; the closest roles are:"
    return _text(
        tool_call_id,
        "agent",
        header
        + "\n"
        + "\n".join(lines)
        + "\n\nlaunch with task(agent='<name>'), or op='list' to see them all.",
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
        installed = install_seed(name, registry=registry)
    except NameTakenError:
        return _error(
            tool_call_id,
            "agent",
            _name_taken_message(name, installing=True),
        )
    if installed is None:
        return _error(tool_call_id, "agent", f"could not install starter {name!r}")
    profile, already_installed = installed
    if already_installed:
        # Say the no-op out loud. `install` is the only verb here that sounds
        # like "put the shipped one back", so answering "installed" to a
        # deliberate skip leaves an operator believing they restored the
        # packaged guidance while their own edited prompt is what the next
        # delegation runs. Same failure shape C4 fixed on the non-role branch.
        return _text(
            tool_call_id,
            "agent",
            f"role {profile.name!r} is already installed and was left as-is (your edits are "
            f"kept): op='show' name={profile.name!r} to read it, or op='update' "
            f"name={profile.name!r} to change it.",
        )
    return _text(
        tool_call_id,
        "agent",
        f"installed role {profile.name!r}: launch with task(agent={profile.name!r}), "
        f"or change what it is told with op='update' name={profile.name!r}.",
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

    # Kind is chosen at CREATE and then frozen. An update that names the other
    # kind is refused rather than converting the row: an ordinary specialist
    # silently acquiring a role's tags (or a role losing them) is a surprising
    # way to lose an agent, which is the fail-open hijack the role tag exists
    # to stop.
    if creating:
        kind = params.kind or "role"
        if existing is not None:
            if kind == "role" and not is_role(existing):
                return _error(tool_call_id, "agent", _name_taken_message(name))
            return _error(
                tool_call_id,
                "agent",
                f"{'role' if is_role(existing) else 'agent'} {name!r} already "
                "exists; use op='update' to change it.",
            )
    else:
        if existing is None:
            return _error(
                tool_call_id,
                "agent",
                f"no registered profile named {name!r} to update.",
            )
        if is_role(existing):
            kind = "role"
            if params.kind == "specialist":
                return _error(
                    tool_call_id,
                    "agent",
                    f"{name!r} is a role; do not pass kind='specialist' to update it.",
                )
        elif any(str(c).strip().lower() == "specialist" for c in (existing.categories or [])):
            kind = "specialist"
            if params.kind == "role":
                return _error(tool_call_id, "agent", _name_taken_message(name))
        else:
            # An ordinary conversational agent is neither a role nor a
            # specialist we authored. Converting it is the fail-open hijack
            # the role tag exists to stop.
            return _error(tool_call_id, "agent", _name_taken_message(name))

    instructions = (params.instructions or "").strip()
    if creating and not instructions:
        return _error(
            tool_call_id, "agent", "create needs 'instructions' — the profile's guidance."
        )
    if len(instructions) > MAX_INSTRUCTIONS_CHARS:
        return _error(
            tool_call_id,
            "agent",
            f"instructions exceed {MAX_INSTRUCTIONS_CHARS} chars; they ride in front of every "
            "run of this profile, so they must stay short.",
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
    tags = list(seed_tags(profile)) if kind == "role" else []
    categories = ["role"] if kind == "role" else ["specialist"]

    # Every field is spelled out (``AgentEditFields`` is validated in strict
    # mode, and every other caller in the tree does the same): a profile inherits
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
            _fields(
                name=name,
                description=profile.description,
                tags=tags,
                categories=categories,
            )
        )
    else:
        agent = existing
        # An update carries only what changed: passing a None description here
        # would blank the routing text a previous create had set.
        overrides: dict[str, Any] = {"tags": tags, "categories": categories}
        if profile.description:
            overrides["description"] = profile.description
        registry.update_agent(agent.id, _fields(**overrides))
    if instructions:
        registry.set_agent_system_prompt(agent.id, instructions)
    verb = "created" if existing is None else "updated"
    if kind == "role":
        how = f"launch with task(agent={name!r})"
    else:
        how = (
            f"launch with --agent {name}, or put it on a team roster; "
            "its instructions are the reusable base, not a team brief"
        )
    return _text(
        tool_call_id,
        "agent",
        f"{verb} {kind} {name!r}; {how}.",
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
            "Reusable agent profiles: delegation roles (reviewer, coder, "
            "architect, manager, designer, scout) and specialists with their "
            "own instruction sets. Find, install, or author one; launch a role "
            "with task(agent='<name>'). A specialist is the reusable base a "
            "team layers collaboration and project briefs on top of."
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
