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
- ``reset`` — put the packaged starter back over an installed role that was
  edited into a bad state, printing the text it replaced.
- ``create`` / ``update`` — author a new role, or fix one whose instructions
  produced a bad run. ``when_to_use`` is stored as the routing description, so
  a role written today is discoverable by ``search`` tomorrow.

INSTALL IS NOT A RESET, AND THAT SEPARATION IS DELIBERATE
=========================================================

``install`` is idempotent: an already-installed role is returned untouched, so
a concurrent second launch of the same role cannot clobber operator edits. That
guarantee holds only because ``install`` is something the harness does FOR the
caller, incidentally, at delegation time — it must never be able to overwrite.
``reset`` is the opposite: an explicit act by someone who has decided they want
the shipped guidance back. It is a separate verb rather than a flag on
``install`` precisely so the safe-by-default op stays safe, and it echoes the
text it replaced so the overwrite is recoverable by copy-paste.

CONTEXT DISCIPLINE
==================

The registry is NEVER enumerated into the prompt: it can be large and its
descriptions are private user content. ``list`` returns one line per profile
and only when called; ``show`` is the only op that returns a full instruction
body. This mirrors how skills are handled — semantic routing decides what is
relevant, and the body loads on demand.
"""

from __future__ import annotations

import difflib
import logging
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from local_operator.agent_profiles import (
    MAX_INSTRUCTIONS_CHARS,
    SEED_ORIGIN_PREFIX,
    AgentProfile,
    NameTakenError,
    install_seed,
    is_role,
    is_specialist,
    list_seeds,
    load_seed,
    matches_seed_text,
    profile_from_agent,
    seed_divergence,
    seed_origin,
    seed_tags,
)
from local_operator.harness.subagent import (
    configured_effort_tiers,
    describe_effort_tiers,
)
from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    AgentToolUpdate,
    ToolContext,
    ToolResult,
)
from local_operator.tools.builtin import (
    _advertise_effort_tiers,
    _error,
    _guard,
    _text,
    _validate_effort_tier,
    _validation_error,
    spill_truncate,
)

logger = logging.getLogger(__name__)


class AgentParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["list", "show", "search", "install", "reset", "create", "update"] = Field(
        description=(
            "search: find a role by meaning; list/show: what exists and what it "
            "says (show also prints the packaged text when an installed role "
            "has diverged from it); install: add a packaged starter; reset: "
            "restore the packaged starter over an edited role, reporting what "
            "it replaced; create/update: author or fix a role or a specialist "
            "profile."
        )
    )
    # The no-spaces guidance is a modularity contract, not registry law: the
    # registry itself accepts any string, but a role/specialist name is also a
    # `/agent <name> <message>` argument parsed at the first whitespace, so a
    # spaced name authors a profile the user cannot invoke from that surface.
    name: str | None = Field(
        default=None,
        description=(
            "Role or specialist name (all ops but search). No spaces — the "
            "name doubles as the /agent slash-command argument."
        ),
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
    # A free string rather than a Literal for the same reason as the ``task``
    # tool's field: a role can only be pinned to a tier the operator has
    # configured, and that set is live config, not something to freeze at
    # import. ``_advertise_effort_tiers`` rewrites the schema to ``inherit``
    # plus the configured tiers; ``_effort_is_configured`` refuses the rest.
    effort: str | None = Field(
        default=None,
        description="create/update: default model tier. 'inherit' clears it.",
    )

    @field_validator("effort", mode="before")
    @classmethod
    def _legacy_empty_effort(cls, value: Any) -> Any:
        """Keep pre-sentinel callers working without advertising invalid schema.

        Empty enum strings are rejected by Gemini function declarations. Older
        callers may still send ``""`` to clear a pin, so normalize that legacy
        wire value before the tier check validates while exposing the
        explicit ``inherit`` spelling to models and providers.
        """

        return "inherit" if value == "" else value

    @field_validator("effort")
    @classmethod
    def _effort_is_configured(cls, value: str | None) -> str | None:
        """A pin must name a tier a launch could honour TODAY.

        Before this, any of ``lo|med|hi`` was accepted and stored, and the pin
        then failed at every launch of the role once the strict path refused
        the unconfigured tier. Refusing here keeps a stale pin from ever being
        written; the launch path still catches one that went stale later.
        """
        if value == "inherit":
            return value
        return _validate_effort_tier(value)

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


def _format_field(value: Any, *, absent: str = "(unset)") -> str:
    """One spelling for a role field in prose, so two readers cannot disagree.

    ``absent`` is overridable because "(unset)" reads as a literal value for a
    field whose values are words: `packaged effort: (unset)` invites the reader
    to parse `(unset)` as a tier.
    """

    if value is None or value == () or value == "":
        return absent
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, tuple):
        return ", ".join(str(item) for item in value)
    return str(value)


def _field_rows(profile: AgentProfile, seed: AgentProfile) -> list[tuple[str, str, str]]:
    """``(field, mine, packaged)`` for the non-prose fields a seed writes.

    Built from one list so the divergence report and the reset echo cannot
    describe different sets of fields.
    """

    no_tier = "not set by the starter"
    return [
        # `description` is the routing text `search` embeds, so replacing it
        # without saying so quietly breaks how the user finds this role.
        (
            "description",
            _format_field(profile.description or profile.when_to_use),
            _format_field(seed.when_to_use or seed.description),
        ),
        ("tools", _format_field(profile.tools), _format_field(seed.tools)),
        (
            "effort",
            _format_field(profile.effort, absent="not set"),
            _format_field(seed.effort, absent=no_tier),
        ),
        ("delegate", _format_field(profile.may_delegate), _format_field(seed.may_delegate)),
    ]


def _instruction_diff(mine: str, packaged: str) -> tuple[str, bool]:
    """``(rendered, is_diff)`` for two instruction bodies.

    The flag is returned rather than inferred by the caller so the heading can
    say which of the two it actually produced: labelling a full body "a diff"
    is a small lie that costs a reader real time.

    Printing both bodies whole was two undelineated ~32-line walls in which a
    reader had to spot a one-line difference by eye, and it is billed to the
    context of whoever asked. ``difflib`` is stdlib, so the common case (a role
    edited in one or two places) collapses to a couple of lines.

    Falls back to the full packaged body when the diff is not actually smaller
    — a wholly rewritten role diffs to something longer than the text itself,
    and a "diff" that exceeds its own input helps nobody.
    """

    packaged_lines = packaged.strip().splitlines()
    diff_lines = list(
        difflib.unified_diff(
            (mine or "").strip().splitlines(),
            packaged_lines,
            fromfile="yours",
            tofile="packaged",
            lineterm="",
            n=1,
        )
    )
    if not diff_lines or len(diff_lines) >= len(packaged_lines):
        return (packaged.strip() or "(none)"), False
    return "\n".join(diff_lines), True


def _may_overwrite(agent: Any, profile: AgentProfile, seed: AgentProfile) -> bool:
    """Whether ``reset`` is allowed to write the seed over this row.

    Two independent ways to establish that the row is a harness copy rather
    than someone's work: an explicit ``seed:`` marker, or prose that is still
    byte-identical to the packaged text. One predicate so ``show`` and
    ``reset`` cannot drift on the question.
    """

    return seed_origin(agent) == seed.name or matches_seed_text(profile, seed)


def _seed_provenance(
    registry: Any, profile: AgentProfile
) -> tuple[AgentProfile, Literal["installed", "unrecorded"]] | None:
    """The packaged seed behind a role, and whether reset may overwrite it.

    Returns None for a role that is not installed (the seed IS what is being
    shown) and for a role with no packaged counterpart. Otherwise it returns
    the seed plus a PROVENANCE verdict, because "which seed is this?" and "may
    reset overwrite it?" are different questions and only ``reset`` needs the
    second one. ``show`` reading the row is not destructive, so gating the
    READ path on provenance is what left a pre-upgrade user at the exact #141
    dead end this tool exists to close.

    The verdict is three-state:

    - ``"installed"`` — the row carries a ``seed:`` marker, or its prose is
      still byte-identical to the packaged text (see
      :func:`matches_seed_text`). Reset may overwrite.
    - ``"unrecorded"`` — a role that predates the marker, or one authored
      under a starter's name. Indistinguishable from each other by
      construction, which is precisely why the caller must NOT claim to know
      which it is. Reset refuses; show still reads.
    """

    if not profile.agent_id:
        return None
    seed = load_seed(profile.name)
    if seed is None:
        return None
    if registry is None:
        return None
    try:
        agent = registry.get_agent_by_name(profile.name)
    except Exception:  # noqa: BLE001 - an unreadable row is not a restore target
        return None
    if agent is None:
        return None
    if _may_overwrite(agent, profile, seed):
        return seed, "installed"
    return seed, "unrecorded"


def _divergent_seed(
    registry: Any, profile: AgentProfile
) -> tuple[AgentProfile, tuple[str, ...], Literal["installed", "unrecorded"]] | None:
    """``(seed, diverged_fields, provenance)`` for a role that no longer matches.

    Divergence itself is decided by :func:`seed_divergence`, the ONE definition
    ``show`` and ``reset`` share. Two independent comparisons is how ``show``
    came to render nothing while a widened allowlist sat in the row.

    Deliberately answers for an ``unrecorded`` row as well. Reading a role is
    not destructive, so the provenance guard has nothing to protect on this
    path — and staying silent here is what left every pre-upgrade user unable
    to see the packaged text, which is the whole of #141. The verdict rides
    along so the caller can offer ``reset`` only where it would actually work.
    """

    found = _seed_provenance(registry, profile)
    if found is None:
        return None
    seed, provenance = found
    diverged = seed_divergence(profile, seed)
    if not diverged:
        return None
    return seed, diverged, provenance


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
    specialists: list[Any] = []
    if registry is not None:
        try:
            specialists = sorted(
                (agent for agent in registry.list_agents() if is_specialist(agent)),
                key=lambda agent: str(agent.name).lower(),
            )
        except Exception:  # noqa: BLE001 — listing is optional enrichment
            specialists = []
    starters = [
        profile
        for profile in (load_seed(name) for name in list_seeds())
        if profile is not None and profile.name.lower() not in installed_names
    ]
    body = ""
    if lines:
        body += "registered roles:\n" + "\n".join(lines)
    if specialists:
        if body:
            body += "\n\n"
        specialist_lines = []
        for agent in specialists:
            summary = (agent.description or "").strip() or "(no description)"
            specialist_lines.append(f"- {agent.name} [specialist]: {summary}")
        body += "registered specialists:\n" + "\n".join(specialist_lines)
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

    registry = _registry(context)
    profile = resolve_profile(name, registry=registry)
    if profile is None and registry is not None:
        try:
            specialist = registry.get_agent_by_name(name)
        except Exception:  # noqa: BLE001
            specialist = None
        if specialist is not None and is_specialist(specialist):
            instructions = registry.get_agent_system_prompt(specialist.id) or "(none)"
            body = (
                f"{specialist.name} — registered specialist\n"
                f"when to use: {specialist.description or '(unstated)'}\n"
                "tools: full inventory\n\n"
                f"instructions:\n{instructions}\n\n"
                f"launch with --agent {specialist.name}, or add it to a team roster."
            )
            text, spill = spill_truncate(body, "agent", context)
            return _text(tool_call_id, "agent", text, details=spill or None)
    if profile is None:
        return _error(
            tool_call_id,
            "agent",
            f"no role or specialist named {name!r} (try op='list')",
        )
    origin = "registered" if profile.agent_id else "packaged starter (not installed)"
    header = [
        f"{profile.name} — {origin}",
        f"when to use: {profile.when_to_use or profile.description or '(unstated)'}",
        f"tools: {', '.join(profile.tools) if profile.tools else 'full inventory'}",
    ]
    if profile.effort:
        header.append(f"effort: {profile.effort}")
    body = "\n".join(header) + "\n\ninstructions:\n" + (profile.instructions or "(none)")
    # Once a packaged role is installed and edited, the SHIPPED guidance is
    # unreachable through every other op: `show` renders the registry copy and
    # `update` asks the reader to retype text they can no longer read. So when
    # the two diverge, print both — the reader's actual complaint is "I cannot
    # see what I lost", and a diff they can copy from answers it without
    # touching their edits. Only the instruction BODY is compared: `when_to_use`
    # is deliberately rewritten on install (the routing text), so comparing the
    # whole profile would call every installed role diverged.
    divergence = _divergent_seed(registry, profile)
    if divergence is not None:
        seed, diverged_fields, _provenance = divergence
        body += "\n\nthis role differs from the packaged starter of the same name ("
        body += ", ".join(diverged_fields) + " differ"
        body += "s" if len(diverged_fields) == 1 else ""
        body += ")."
        if "instructions" in diverged_fields:
            # WHAT the reader is told to do decides what they are shown. On a
            # resettable row `reset` does the applying, so a diff is the useful
            # rendering: it answers "what changed?" in a couple of lines. On an
            # unrecorded row the closing line asks them to apply the values
            # THEMSELVES, and a diff is not appliable — it carries only the
            # changed lines, so the instruction and the output disagreed and
            # the op that prints the full body was the one they were steered
            # away from. Print what they are being asked to copy.
            if _provenance == "installed":
                rendered, is_diff = _instruction_diff(profile.instructions or "", seed.instructions)
                label = (
                    "packaged instructions, as a diff against yours"
                    if is_diff
                    else "packaged instructions in full (too different to diff usefully)"
                )
            else:
                rendered = seed.instructions.strip() or "(none)"
                label = "packaged instructions in full"
            body += f"\n{label}:\n{rendered}"
        for field, mine, packaged in _field_rows(profile, seed):
            if field in diverged_fields:
                body += f"\npackaged {field}: {packaged}  (yours: {mine})"
    # The highest-intent moment in the flow: someone has just read a role's
    # guidance and decided they want it. Every other op ends by naming the next
    # command; this one used to stop at "not installed" and leave them there.
    body += f"\n\nlaunch with task(agent={profile.name!r})"
    if not profile.agent_id:
        body += f", or op='install' name={profile.name!r} to edit it"
    elif divergence is not None and divergence[2] == "installed":
        body += f", or op='reset' name={profile.name!r} to restore the packaged version"
    elif divergence is not None:
        # An unrecorded row cannot be reset (reset will not overwrite what it
        # cannot prove the harness wrote), so offering the verb here would be
        # the same dead end #141 was filed about. Name the op that DOES work
        # on the values printed above, which is what makes them copy-pasteable.
        body += f", or op='update' name={profile.name!r} to apply the packaged values yourself"
    body += "."
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
        # Offer `reset` ONLY where it would work. Advertising it on a row reset
        # will refuse is the same dead end #141 was filed about, relocated into
        # a second message: `show` was gated on provenance and this was not, so
        # a legacy edited row was still sent to a verb that declines. One
        # predicate decides it everywhere.
        found = _seed_provenance(registry, profile)
        resettable = found is not None and found[1] == "installed"
        restore = (
            f"op='reset' name={profile.name!r} to put the packaged version back, or "
            if resettable
            else ""
        )
        return _text(
            tool_call_id,
            "agent",
            f"role {profile.name!r} is already installed and was left as-is (your edits are "
            f"kept): op='show' name={profile.name!r} to read it (it prints the packaged "
            f"version too when yours has diverged), "
            f"{restore}op='update' name={profile.name!r} to change it.",
        )
    return _text(
        tool_call_id,
        "agent",
        f"installed role {profile.name!r}: launch with task(agent={profile.name!r}), "
        f"or change what it is told with op='update' name={profile.name!r}.",
    )


async def _op_reset(context: ToolContext | None, tool_call_id: str, name: str) -> ToolResult:
    """Restore a packaged seed over an installed role, echoing what it replaced.

    Separate from ``install`` on purpose. ``install`` runs incidentally (a
    delegation asking for a role the operator has never created materializes
    it), so it must be unable to overwrite: its idempotence is what stops a
    concurrent second launch from clobbering operator edits. ``reset`` is only
    ever reached because a caller typed the word, which is the consent
    ``install`` cannot infer. The one property the two share is that neither
    may destroy work silently, so this prints the replaced instructions back:
    the overwrite stays recoverable by copy-paste even though the registry now
    holds the packaged text.
    """

    registry = _registry(context)
    if registry is None:
        return _error(
            tool_call_id, "agent", "no agent registry attached to this session; cannot reset."
        )
    seed = load_seed(name)
    if seed is None:
        # A role with no packaged counterpart has nothing to be restored TO, so
        # resetting it could only mean deleting it, which is never what the word
        # means here. Name the command that DOES work for each of the two ways a
        # reader lands here: an existing role of theirs (edit it) versus a name
        # that is simply not a starter (list what is). Sending them to op='list'
        # alone was a second dead end for the first group.
        registered = ""
        try:
            if registry.get_agent_by_name(name) is not None:
                registered = (
                    f" You do have a role named {name!r}: it was authored here rather than "
                    f"installed, so edit it with op='update' name={name!r}."
                )
        except Exception:  # noqa: BLE001 - enrichment only; the refusal still stands
            registered = ""
        return _error(
            tool_call_id,
            "agent",
            f"no packaged starter named {name!r}, so there is nothing to reset it to; "
            f"nothing was changed.{registered} Starters that can be reset: "
            f"{', '.join(list_seeds())}.",
        )

    try:
        existing = registry.get_agent_by_name(seed.name)
    except Exception:  # noqa: BLE001 - a lookup failure must not overwrite blindly
        logger.warning("agent registry lookup failed for %r", seed.name)
        existing = None
    # `install_seed(overwrite=True)` skips its own non-role guard, by design:
    # the kwarg means "the caller has decided". That makes checking here
    # load-bearing rather than defensive — without it, resetting a name that
    # happens to belong to an ordinary conversational agent would rewrite that
    # agent's system prompt with role guidance.
    if existing is not None and not is_role(existing):
        return _error(
            tool_call_id,
            "agent",
            # Name the escape that actually RUNS. This used to say "install the
            # starter under a different name", which cannot work — `install`
            # only accepts packaged names and nothing renames — leaving a
            # history-destroying shell delete as the only route on the one path
            # that has a safe in-tool answer. `show` renders the packaged body
            # even when the name is occupied, and `create` accepts any name, so
            # read-then-author is the recovery, and the destructive option is
            # mentioned last with its cost.
            f"an agent named {seed.name!r} already exists and is not a role, so nothing was "
            f"reset. To get the packaged guidance under a name you can use: op='show' "
            f"name={seed.name!r} to read it, then op='create' name='my-{seed.name}' with those "
            f"instructions. To free the name itself, `local-operator agents delete --name "
            f"{seed.name}` at the shell deletes that agent and its conversation history; this "
            "tool cannot delete or rename agents.",
        )

    previous = profile_from_agent(registry, existing) if existing is not None else None
    diverged = seed_divergence(previous, seed) if previous is not None else ()
    # PROVENANCE, not name collision. A role the operator authored under a
    # starter's name has a seed but is not a copy OF it, so restoring "the
    # packaged version" over it would delete work the harness never wrote —
    # while the name-only check that used to guard this reported success.
    #
    # The marker is only one of two ways to establish that. A row whose prose
    # still matches the seed byte-for-byte holds nobody's words, so it unlocks
    # even without a marker; without that escape hatch every role installed by
    # an earlier release would be permanently refused, which turned a safety
    # guard into the very #141 dead end this tool exists to close.
    if (
        existing is not None
        and previous is not None
        and not _may_overwrite(existing, previous, seed)
    ):
        # Do NOT assert authorship. Absence of a marker is not evidence that a
        # human wrote this: a pre-marker install and a hand-authored role are
        # indistinguishable here BY CONSTRUCTION, so claiming the second is a
        # positive statement about history the tool cannot support and the
        # reader cannot falsify. Say what is actually known — there is no
        # install record — and hand over the packaged values so the reader can
        # apply them through an op that exists.
        lead = (
            f"role {seed.name!r} has no record of being installed from the packaged starter, "
            f"so reset will not overwrite it and nothing was changed. Rows created before this "
            f"record existed look the same as a role you wrote yourself under a starter's name."
        )
        # U8 is answered by the unlock rather than by wording here: a row that
        # already matches the seed has matching prose by definition, so it is
        # treated as installed and takes the ordinary "already matches" path
        # above. Reaching this point therefore always means real divergence,
        # and a "nothing to change" arm here would be unreachable.
        body = lead + "\n\nthe packaged values, to apply yourself:"
        if "instructions" in diverged:
            body += f"\n\npackaged instructions:\n{seed.instructions.strip()}\n"
        for field, mine, packaged in _field_rows(previous, seed):
            if field in diverged:
                body += f"\npackaged {field}: {packaged}  (yours: {mine})"
        body += (
            f"\n\nop='update' name={seed.name!r} to apply them, or op='show' "
            f"name={seed.name!r} to read the whole role first."
        )
        text, spill = spill_truncate(body, "agent", context)
        return _text(tool_call_id, "agent", text, details=spill or None)
    if previous is not None and not diverged:
        # Reporting a restore for a no-op is the exact misreport `install`'s
        # message was fixed for; the answer to "did it change?" has to be no.
        # This compares EVERY field the seed writes, via the same predicate
        # `show` uses: comparing instructions alone made the restore
        # unreachable for a role whose allowlist had been widened but whose
        # prose was untouched, so the one case the overwrite fix exists for
        # returned "nothing was changed" and kept the widened tool surface.
        return _text(
            tool_call_id,
            "agent",
            f"role {seed.name!r} already matches the packaged version; nothing was changed.",
        )

    try:
        installed = install_seed(seed.name, registry=registry, overwrite=True)
    except NameTakenError:  # pragma: no cover - guarded above, kept for safety
        return _error(tool_call_id, "agent", _name_taken_message(seed.name, installing=True))
    if installed is None:
        return _error(tool_call_id, "agent", f"could not reset role {seed.name!r}")
    profile, _ = installed

    if previous is None:
        # Reset on a starter that was never installed is not an error: the end
        # state the caller asked for (registry holds the packaged text) is the
        # one they got. Saying which of the two happened keeps the report true.
        body = (
            f"role {profile.name!r} was not installed; installed the packaged version. "
            f"launch with task(agent={profile.name!r})."
        )
        return _text(tool_call_id, "agent", body)
    # Echo EVERY field that was replaced, not just the prose. The restore is
    # only "recoverable by copy-paste" for what it prints, and a role someone
    # had deliberately NARROWED to `read` used to lose that restriction with no
    # record at all — while the closing line pointed them at op='update' for the
    # instructions, the one thing that HAD been echoed. Now that non-prose
    # divergence actually reaches the overwrite, silence here would start
    # discarding tool restrictions the old short-circuit happened to leave
    # alone, so the report has to widen with the fix.
    body = f"reset role {profile.name!r} to the packaged version ("
    body += ", ".join(diverged) + " replaced).\n"
    if "instructions" in diverged:
        body += f"\nyour instructions were:\n{(previous.instructions or '(none)').strip()}\n"
    for field, mine, _packaged in _field_rows(previous, seed):
        if field in diverged:
            body += f"\nyour {field}: {mine}"
    body += (
        f"\n\nlaunch with task(agent={profile.name!r}), or op='update' "
        f"name={profile.name!r} to put your own settings back."
    )
    text, spill = spill_truncate(body, "agent", context)
    return _text(tool_call_id, "agent", text, details=spill or None)


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
        elif is_specialist(existing):
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

    if params.effort == "inherit":
        effort = None
    elif params.effort is not None:
        effort = params.effort
    else:
        # Omission and JSON null both mean "leave unchanged"; only the explicit
        # sentinel may remove a stored pin, so a model cannot clear configuration
        # merely by filling an optional nullable field with its default.
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
    # PRESERVE provenance across an edit. `seed_tags` encodes the profile's own
    # fields and knows nothing about where the row came from, so rebuilding from
    # it alone silently un-marks an installed role the moment anyone edits it —
    # which would make `reset` refuse to restore exactly the roles it exists for
    # (an edited install is its entire motivating case). An edit changes what a
    # role SAYS, never where it came from.
    if kind == "role" and existing is not None:
        origin = seed_origin(existing)
        if origin:
            tags.append(f"{SEED_ORIGIN_PREFIX}{origin}")
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
    """Discover, install, reset, and author reusable agent role profiles.

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

    if (
        params.op in {"show", "install", "reset", "create", "update"}
        and not (params.name or "").strip()
    ):
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
    if params.op == "reset":
        return await _op_reset(context, tool_call_id, str(params.name))
    return await _op_write(context, tool_call_id, params, creating=params.op == "create")


def _effort_pin_description() -> str:
    """The ``effort`` description for create/update, matching the live schema.

    With tiers configured it names what each resolves to so a role is pinned
    on information; with none it says so and that ``inherit`` (the only
    member left in the enum) is the whole choice. Short: billed every turn.
    """
    tiers = configured_effort_tiers()
    if not tiers:
        return (
            "create/update: no model tiers are configured (values.subagents.models); "
            "'inherit' clears a pin and every role inherits the launching session's model."
        )
    return (
        f"create/update: default model tier ({describe_effort_tiers(tiers)}). "
        "'inherit' clears it."
    )


def build_agent_tool(context: ToolContext) -> AgentTool | None:
    """createIf: the tool exists only where a registry can back it.

    Without a registry the read-only ops would still work off packaged
    starters, but ``install``/``reset``/``create``/``update`` could not persist anything
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
            "own instruction sets. Find, install, author, or reset one to its "
            "packaged version; launch a role with task(agent='<name>'). A "
            "specialist is the reusable base a team layers collaboration and "
            "project briefs on top of."
        ),
        parameters=_advertise_effort_tiers(
            AgentParams.model_json_schema(),
            description=_effort_pin_description(),
            extra=("inherit",),
        ),
        # Writes land in the user's own configuration directory, never in the
        # workspace, and are trivially reversible by editing the profile back.
        # Gating them behind an approval prompt would make an agent improving
        # its own reviewer guidance an interruption, which is exactly the
        # friction that keeps the registry empty. ``reset`` stays inside that
        # reasoning rather than escaping it: it is the one op that overwrites
        # instructions the operator wrote, so it prints them back in its result
        # and an op='update' with that text restores the prior state exactly.
        approval_tier="read",
        concurrency="exclusive",
        interruptible=False,
        execute=execute_agent,
    )
