"""Agent PROFILES: the role half of a registered agent, and how a role is chosen.

WHY THIS EXISTS
===============

Delegation used to carry a role only in the prose a parent happened to write
into a ``task`` prompt. Measured over 99 sessions in one day, 58 were review
children, and the prompts hand-written for them ranged from 161 to 9551
characters — a 59x spread for what is the same job every time. Everything that
makes a review fast (read the diff before the tree, classify by severity, stop
when nothing blocking is left, do not rewrite the code you are reviewing) had
to be re-derived by the parent on every launch, and whatever it forgot, the
child did not do.

The fix is NOT a hardcoded table of roles in the harness. Roles are user data:
which ones exist, what they say, and when they apply all differ per operator
and per repository, and they must be editable by the person — or the agent —
who notices the guidance was wrong. So a role is a REGISTERED AGENT
(:mod:`local_operator.agents`), which already persists a name, description,
tags, model, sampling settings, and a ``system_prompt.md``. This module adds
only what the registry lacked:

1. **Applicability** — ``when_to_use`` on the profile, so a row can say what it
   is FOR rather than leaving an embedder to infer it from a name. It rides in
   the same semantic index that already routes skills and guides, so choosing
   an agent costs no extra context: the registry is never enumerated into the
   prompt (see :func:`local_operator.session_factory._registered_agent_hints`).

2. **A tool surface** — ``tools`` on the profile, so a reviewer physically
   cannot push a "helpful" fix (the failure that forces a re-review of a diff
   the reviewer itself changed). Enforced at child construction.

3. **Seeds** — a small set of packaged starter profiles (reviewer, coder,
   architect, manager, designer, scout) written as plain markdown with
   frontmatter, installed into the user's registry ON DEMAND. They are a
   starting point the operator owns and edits, not a fixed enum: after install
   they are ordinary registry rows with no special status, and a task can
   equally use a profile the operator (or an agent, via the ``agent`` tool)
   wrote from scratch.

The seeds ship as files rather than string constants so that "read what the
reviewer is told" and "change what the reviewer is told" are the same
operation for a human and for an agent.

TOKEN BUDGET
============

A profile's instructions are prepended to the child's prompt, so every line is
billed on each launch of that role and again on every turn of that child's own
loop. Seed bodies are therefore short and imperative: a line earns its place by
changing what the child DOES, not by describing good practice in general. The
seed catalogue itself never enters a prompt — it is discovered semantically and
only the selected row's body is loaded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence, TypeVar

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.agents import AgentData, AgentRegistry

logger = logging.getLogger(__name__)

#: Where the packaged starter profiles live (one ``<name>.md`` per role).
SEEDS_DIR = Path(__file__).parent / "agent_seeds"

#: Tag prefix recording that a role row was WRITTEN FROM a packaged seed, e.g.
#: ``seed:reviewer``. The registry is a flat namespace shared with roles the
#: operator authored themselves, so without this marker a row named ``scout``
#: that someone wrote from scratch is byte-indistinguishable from an installed
#: ``scout`` — both are just ``['role', ...]``. ``reset`` is destructive, so it
#: needs to know which of the two it is looking at; see :func:`seed_origin`.
#:
#: Deliberately NOT part of :func:`seed_tags`: that function encodes a
#: PROFILE's fields, and provenance is a property of the registry ROW, not of
#: the profile. Folding it in would make ``op='create'`` stamp a self-authored
#: role as seed-derived just because it renders the same fields.
SEED_ORIGIN_PREFIX = "seed:"

#: Cap on an instruction body admitted from a profile. A profile is user data
#: and rides in front of a child's prompt on every turn, so an unbounded body
#: is an unbounded per-turn bill. Generous enough for a detailed role brief,
#: bounded enough that a pasted log cannot become a permanent tax.
MAX_INSTRUCTIONS_CHARS = 8_000

#: The tag that marks a registry row as a delegation ROLE. A registry also
#: holds ordinary conversational agents and autosave rows in the same flat
#: namespace, keyed by user-visible name, so without this marker any agent that
#: merely happened to be called ``reviewer`` would be launched AS the reviewer
#: role — with no allowlist, and therefore the full write inventory, while the
#: child was still told it was a reviewer. Written by seed installation and by
#: the ``agent`` tool; required by :func:`resolve_profile`.
ROLE_TAG = "role"

#: The tools a read-only role may reach the NETWORK with, and the floor every
#: allowlisted role keeps (see
#: :func:`local_operator.harness.subagent._with_network_floor`). Both are
#: ``approval_tier="read"``: they retrieve a remote document and produce no
#: side effect beyond a bounded cache under ``config_dir()``, which is the same
#: promise ``read`` makes about the disk.
#:
#: They are named separately from :data:`READ_ONLY_TOOLS` because the two lists
#: answer different questions. ``READ_ONLY_TOOLS`` is "what does this role
#: start with"; this tuple is "what can never be taken away from it", and a
#: registry row written before these tools existed has to be repaired against
#: the second list, not the first.
READ_ONLY_NETWORK_TOOLS = ("web_search", "web_fetch")

#: Tool names a read-only role is filtered to. Allowlist, not a tier filter:
#: approval tiers drift as tools are added, and these roles promise no LOCAL
#: side effects, which is narrower than "nothing marked write". ``browser``
#: drives the user's real browser and ``eval`` executes code, so both are
#: excluded by NAME even where a tier check alone would admit them.
#:
#: Read-only means "changes nothing", NOT "reaches nothing". Omitting the
#: network tools here was a capability bug, not a safety property: a ``scout``
#: launched to research a question on the web reported "I have no network
#: access in this session" and fell back to grepping the local disk for facts
#: that were never on it, burning its whole budget to produce nothing. A role
#: whose entire purpose is research cannot be structurally incapable of it, and
#: retrieving a page mutates no more than reading a file does.
READ_ONLY_TOOLS = (
    "read",
    "glob",
    "grep",
    "list_variables",
    "read_variable",
) + READ_ONLY_NETWORK_TOOLS


class NameTakenError(RuntimeError):
    """An agent of that name exists but is not a role.

    Its own class rather than a bare ``RuntimeError`` so the caller can tell
    "this name is occupied by something else" apart from a registry failure,
    and say so instead of reporting a successful install that wrote nothing.
    """

    def __init__(self, name: str) -> None:
        super().__init__(f"an agent named {name!r} exists and is not a role")
        self.name = name


@dataclass(frozen=True)
class AgentProfile:
    """A role resolved from the registry (or from a packaged seed).

    ``tools=None`` means "whatever the parent would build"; a non-empty tuple
    filters the child's inventory to exactly those names. ``effort`` is the
    default model tier for the role and is always overridable per launch,
    because the right model for a role depends on the operator's provider mix
    rather than on anything this file can know. The packaged seeds pin NO
    tier: a child inherits the session's model unless the operator (or the
    launch) picks one, so a review round never silently lands on a weaker
    model than the session it is checking.
    """

    name: str
    description: str = ""
    when_to_use: str = ""
    instructions: str = ""
    tools: tuple[str, ...] | None = None
    effort: str | None = None
    #: Whether a child in this role may delegate further. A reviewer that
    #: spawns its own children turns one review into a fan-out nobody is
    #: watching; a read-only role that delegates autonomous work is not
    #: read-only.
    may_delegate: bool = False
    #: Registry id when this profile came from a registered agent, else None
    #: for a packaged seed resolved without installing it.
    agent_id: str | None = None
    #: Provider/model selector (``provider/model-id``) the profile pins, if any.
    model: str = ""
    hosting: str = ""

    @property
    def preamble(self) -> str:
        """The text stamped in front of a child's prompt for this role.

        Empty instructions yield an empty preamble rather than a header with
        nothing under it — a role that says nothing must cost nothing.
        """

        body = self.instructions.strip()
        if not body:
            return ""
        return f"[role: {self.name}]\n{body}\n\n"


def _split_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Return ``(frontmatter, body)`` for a seed/profile markdown file.

    Reuses the skills frontmatter parser so a profile file and a SKILL.md are
    parsed by exactly one implementation; a second YAML-ish parser beside it is
    how the two would later disagree about the same bytes.
    """

    from local_operator.skills.discovery import parse_frontmatter

    meta = parse_frontmatter(text)
    if not text.startswith("---"):
        return meta, text
    lines = text.split("\n")
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return meta, "\n".join(lines[i + 1 :]).strip()
    return meta, ""


def _as_tuple(raw: object) -> tuple[str, ...] | None:
    """Normalize a ``tools`` frontmatter value to a tuple, or None.

    Accepts a YAML list or a comma-separated string, because both are what a
    human writes and neither is worth an error. An explicit empty list is
    treated as "unset": a child with zero tools cannot do anything, so it is
    always a mistake rather than an intent.
    """

    if raw is None:
        return None
    if isinstance(raw, str):
        names = [part.strip() for part in raw.split(",")]
    elif isinstance(raw, (list, tuple)):
        names = [str(part).strip() for part in raw]
    else:
        return None
    names = [name for name in names if name]
    return tuple(dict.fromkeys(names)) or None


def _profile_from_text(name: str, text: str, *, agent_id: str | None = None) -> AgentProfile:
    """Build a profile from markdown with frontmatter."""

    meta, body = _split_frontmatter(text)

    def _str(key: str) -> str:
        value = meta.get(key)
        return str(value).strip() if value is not None else ""

    # ``delegate`` and ``may_delegate`` both work: the tag encoding on a
    # registered profile spells it ``delegate:yes``, and a human editing a seed
    # file should not have to remember which of the two spellings this parser
    # happens to prefer.
    delegate_raw = meta.get("delegate", meta.get("may_delegate", False))
    if isinstance(delegate_raw, str):
        may_delegate = delegate_raw.strip().lower() in {"1", "true", "yes", "on"}
    else:
        may_delegate = bool(delegate_raw)

    return AgentProfile(
        name=_str("name") or name,
        description=_str("description"),
        when_to_use=_str("when_to_use"),
        instructions=body[:MAX_INSTRUCTIONS_CHARS],
        tools=_as_tuple(meta.get("tools")),
        effort=_str("effort") or None,
        may_delegate=may_delegate,
        agent_id=agent_id,
        model=_str("model"),
        hosting=_str("hosting"),
    )


# -- packaged seeds ---------------------------------------------------------


def list_seeds() -> list[str]:
    """Names of the packaged starter profiles, deterministically ordered.

    Never raises: a missing or unreadable seeds directory means "no starters
    available", which degrades to the operator writing their own, not to a
    failed session.
    """

    try:
        return sorted(path.stem for path in SEEDS_DIR.glob("*.md"))
    except OSError:  # pragma: no cover - unreadable package dir
        logger.warning("agent seed catalogue unreadable at %s", SEEDS_DIR)
        return []


def load_seed(name: str) -> AgentProfile | None:
    """Read one packaged seed by name, or None when it does not exist.

    The name is resolved against the catalogue rather than joined onto the
    directory, so a caller passing ``../../etc/passwd`` gets None instead of a
    path traversal.
    """

    key = (name or "").strip().lower()
    if not key or key not in set(list_seeds()):
        return None
    try:
        text = (SEEDS_DIR / f"{key}.md").read_text(encoding="utf-8", errors="replace")
    except OSError:
        logger.warning("agent seed %s could not be read", key)
        return None
    return _profile_from_text(key, text)


def seed_catalogue() -> list[AgentProfile]:
    """Every packaged seed, for listing and for semantic routing."""

    return [profile for profile in (load_seed(name) for name in list_seeds()) if profile]


# -- registry-backed profiles ----------------------------------------------


def _agent_instructions(registry: "AgentRegistry", agent: "AgentData") -> str:
    """The agent's own ``system_prompt.md``, bounded, or ''.

    Never raises: a profile whose instructions cannot be read still names a
    valid role with a valid tool surface, and losing the delegation over a
    read error would be a worse outcome than running it with less guidance.
    """

    try:
        return (registry.get_agent_system_prompt(agent.id) or "")[:MAX_INSTRUCTIONS_CHARS]
    except Exception:  # noqa: BLE001 - guidance is best-effort
        logger.warning("could not read instructions for agent %s", agent.id)
        return ""


def is_role(agent: "AgentData") -> bool:
    """Whether a registry row is marked as a delegation role.

    One implementation, because two readers that disagree about what a role is
    are how ``op='list'`` came to hide a row that ``task(agent=...)`` would
    happily run.
    """

    return any(str(tag).strip().lower() == ROLE_TAG for tag in (agent.tags or []))


def is_specialist(agent: "AgentData") -> bool:
    """Whether the agent tool authored this row as a reusable specialist.

    Ordinary conversational agents share the registry and must stay private to
    ``agent list/show``. The category is the explicit marker written by
    ``agent(op='create', kind='specialist')``; a name or a non-empty prompt is
    not enough, because both are normal on private chat agents too.
    """

    return any(
        str(category).strip().lower() == "specialist" for category in (agent.categories or [])
    )


def profile_from_agent(registry: "AgentRegistry", agent: "AgentData") -> AgentProfile:
    """Convert a registered agent into a role profile.

    ``when_to_use`` and the tool surface are carried in the agent's tags with a
    ``key:value`` shape (``tools:read,grep`` / ``effort:lo`` / ``delegate:yes``)
    because ``AgentData`` is a persisted, API-exposed model whose schema is
    shared with the server routes and desktop UI: encoding two optional role
    fields in tags keeps every existing profile, export archive, and client
    valid, where adding columns would force a migration on all of them. The
    ``when_to_use`` prose itself rides in ``description``, which is already the
    semantic routing text.
    """

    tags = [str(tag) for tag in (agent.tags or [])]
    tools: tuple[str, ...] | None = None
    effort: str | None = None
    may_delegate = False
    for tag in tags:
        key, sep, value = tag.partition(":")
        if not sep:
            continue
        key = key.strip().lower()
        value = value.strip()
        if key == "tools":
            tools = _as_tuple(value)
        elif key == "effort":
            effort = value or None
        elif key == "delegate":
            may_delegate = value.lower() in {"1", "true", "yes", "on"}

    return AgentProfile(
        name=agent.name,
        description=str(agent.description or ""),
        when_to_use=str(agent.description or ""),
        instructions=_agent_instructions(registry, agent),
        tools=tools,
        effort=effort,
        may_delegate=may_delegate,
        agent_id=agent.id,
        model=str(agent.model or ""),
        hosting=str(agent.hosting or ""),
    )


def resolve_profile(
    name: str | None,
    *,
    registry: Any = None,
) -> AgentProfile | None:
    """Resolve a role NAME to a profile: registry first, then packaged seeds.

    Registry first is the whole point of making these editable — once an
    operator has a ``reviewer`` of their own, theirs is the one that runs, and
    the packaged seed of the same name becomes irrelevant rather than
    competing with it.

    Returns None for an unknown name. The caller decides what that means:
    ``task`` treats it as "no role" and launches a full child rather than
    failing, because the parent already decided the work should happen and a
    typo in a role name is not a reason to lose the delegation.

    ``registry`` is typed ``Any`` rather than ``AgentRegistry``: it is reached
    through ``getattr`` off a session (whose host may attach any object that
    answers the two methods used here), and every access below is already
    guarded, so a narrower annotation would claim a coupling the code does not
    actually require.
    """

    key = (name or "").strip()
    if not key:
        return None
    if registry is not None:
        try:
            agent = registry.get_agent_by_name(key)
            if agent is not None and not is_role(agent):
                # An exact match that is NOT a role must not end the search.
                # It used to, which reopened the very bug the fold was added
                # for: with an ordinary agent named `reviewer` beside the
                # operator's own `Reviewer` role, the exact hit was discarded
                # as a non-role and the fold never ran, so the packaged seed
                # silently shadowed the operator's role.
                agent = None
            if agent is None:
                # Case-insensitive retry, because ``load_seed`` folds case and
                # an exact-only registry lookup would invert this function's
                # whole point: ``task(agent="Reviewer")`` would find the
                # PACKAGED seed while silently ignoring the operator's own
                # ``Reviewer``. An exact ROLE match still wins; this only runs
                # when the exact lookup yielded no role.
                folded = key.casefold()
                agent = next(
                    (
                        row
                        for row in registry.list_agents()
                        if str(row.name).strip().casefold() == folded and is_role(row)
                    ),
                    None,
                )
        except Exception:  # noqa: BLE001 - registry problems must not fail a launch
            agent = None
            logger.warning("agent registry lookup failed for %r", key)
        # The row must be MARKED a role. An unmarked same-named agent falls
        # through to the packaged seed rather than being run as the role: the
        # registry is a flat namespace of user-visible names shared with
        # ordinary chat agents, and honouring one of those would hand a child
        # the full write inventory under a role's name.
        if agent is not None and is_role(agent):
            return profile_from_agent(registry, agent)
    return load_seed(key)


def resolve_profile_or_specialist(
    name: str | None,
    *,
    registry: Any = None,
) -> tuple[str | None, "AgentProfile | None", str, str]:
    """Resolve a NAME to an attachable persona, priority order fixed HERE.

    The SINGLE source of truth for how a name becomes a persona, shared by
    ``/agent`` attach, a team's manager resolution, AND the org-chart resolver
    (:func:`local_operator.org_chart.resolve_org`). Three callers, one order,
    so they can never disagree about which of a role, a specialist, and a
    packaged seed wins — the A1 bug and its team twin were exactly that
    disagreement, and a classifier reimplemented beside this one is how it
    would come back.

    Order, strongest first:

    1. the operator's own registered ROLE — ``resolve_profile`` returns a
       profile with a non-``None`` ``agent_id`` only for a real registry role
       (never a packaged seed), so an ``agent_id`` here is the operator's own
       role and outranks everything below;
    2. the operator's own SPECIALIST — checked BEFORE the seed fallthrough,
       which is the whole fix: ``resolve_profile`` honours only role rows and
       otherwise returns the SEED, so a specialist named after a seed word
       would otherwise be shadowed by that seed;
    3. a packaged SEED resolved by ``resolve_profile`` (``agent_id`` is
       ``None``), so ``reviewer`` and friends still resolve on a fresh machine
       with no registry row of that name.

    Returns ``(kind, profile, specialist_prompt, display_name)`` where ``kind``
    is ``"role"``/``"seed"`` (``profile`` set, ``specialist_prompt`` empty),
    ``"specialist"`` (``profile`` ``None``, ``specialist_prompt`` set), or
    ``None`` (nothing attachable by that name — ``profile`` ``None`` and both
    strings empty). Ordinary conversational/autosave rows are not attachable:
    only an explicit ``is_specialist`` marker or a role tag qualifies, so a
    private chat agent's prompt is never pulled in by a coincidental name.

    ``registry`` is typed ``Any`` for the same reason as ``resolve_profile``:
    it is reached through ``getattr`` off a session, whose host may attach any
    object answering the two methods used here, and every access is guarded.
    """

    key = (name or "").strip()
    if not key:
        return (None, None, "", "")
    profile = resolve_profile(key, registry=registry)
    if profile is not None and profile.agent_id is not None:
        return ("role", profile, "", profile.name)
    if registry is not None:
        try:
            specialist = registry.get_agent_by_name(key)
            if specialist is not None and is_specialist(specialist):
                prompt = (registry.get_agent_system_prompt(specialist.id) or "").strip()
                return ("specialist", None, prompt, str(specialist.name))
        except Exception:  # noqa: BLE001 - registry problems mean "not found"
            pass
    if profile is not None:
        return ("seed", profile, "", profile.name)
    return (None, None, "", "")


def classify_name(
    name: str | None, *, registry: Any = None
) -> Literal["role", "specialist", "seed", "unresolved"]:
    """The KIND half of :func:`resolve_profile_or_specialist`, for the org chart.

    Returns ``"role"`` / ``"specialist"`` / ``"seed"`` / ``"unresolved"`` for a
    member name. The org-chart resolver only needs to tag WHAT a leaf is, not
    to attach its instructions, so it calls this thin wrapper rather than
    carrying the whole persona tuple — but the wrapper delegates to the ONE
    resolver above so the chart's tag can never drift from what an attach would
    actually pick. ``None`` from the resolver (nothing attachable) reads as
    ``"unresolved"`` here: a name that matches nothing renders as a dim ghost.
    """

    kind, _profile, _prompt, _display = resolve_profile_or_specialist(name, registry=registry)
    # The resolver's ``kind`` is one of exactly these four labels or ``None``
    # (nothing attachable) — the latter reads as "unresolved" for the chart.
    if kind in ("role", "specialist", "seed"):
        return kind  # type: ignore[return-value]
    return "unresolved"


def install_seed(
    name: str,
    *,
    registry: "AgentRegistry",
    overwrite: bool = False,
) -> tuple[AgentProfile, bool] | None:
    """Copy a packaged seed into the registry; return ``(profile, already_installed)``.

    The second element is why this returns a tuple: the caller has to be able
    to tell "I installed it" from "it was already there and I left it alone",
    because reporting the first when the second happened misleads an operator
    who is trying to restore a role they broke.

    This is the "readily available, pulled in as needed" step: seeds are not
    installed at startup (an empty registry should stay empty until something
    needs a role), so the first delegation that asks for a role the operator
    has never created materializes it here, once, as an ordinary editable
    registry row.

    Idempotent: an existing ROLE of the same name is returned untouched unless
    ``overwrite`` is set, so a concurrent second launch of the same role cannot
    duplicate the profile or clobber edits the operator has made.

    ``overwrite`` restores exactly the fields the SEED owns — instructions,
    routing description, and the role tags carrying the tool allowlist, effort
    and delegate flag. It deliberately does not touch ``model``, ``hosting``,
    ``security_prompt`` or the sampling settings: those are the operator's, a
    seed pins none of them, and ``update_agent`` skips ``None`` values so they
    cannot be cleared through this path anyway. A reset is therefore "the
    packaged ROLE back", not "a factory-reset row" — worth stating because
    ``security_prompt`` in particular survives one. It also bypasses the
    :class:`NameTakenError` guard,
    because the kwarg means "the caller has already decided". Both properties
    make it unsafe to reach on an incidental install path: it is exposed to
    users only through the ``agent`` tool's explicit ``op='reset'``, which does
    its own non-role check and echoes the instructions it replaced.

    Raises :class:`NameTakenError` when the name belongs to an agent that is
    NOT a role. Returning that row (which is what this used to do) reported a
    successful install while writing nothing, so an operator recovering from a
    misbehaving role was told the fix had landed when it had not.
    """

    seed = load_seed(name)
    if seed is None:
        return None

    from local_operator.agents import AgentEditFields

    existing = None
    try:
        existing = registry.get_agent_by_name(seed.name)
    except Exception:  # noqa: BLE001
        existing = None
    if existing is not None and not is_role(existing) and not overwrite:
        raise NameTakenError(seed.name)
    if existing is not None and not overwrite:
        # Already a role: return it UNTOUCHED (that idempotence is what keeps a
        # concurrent second launch from clobbering operator edits) and say so
        # via ``already_installed``, so the caller does not report a write that
        # did not happen. The natural recovery guess after breaking a role is
        # "install it again", and answering "installed" to a no-op leaves the
        # operator believing the packaged guidance is back when their own
        # edited prompt is what the next delegation will run.
        return profile_from_agent(registry, existing), True

    # The provenance marker rides alongside the profile's own field tags. It is
    # what later lets ``reset`` tell an installed copy from a role the operator
    # authored under a name that happens to collide with a starter.
    tags = [*seed_tags(seed), f"{SEED_ORIGIN_PREFIX}{seed.name}"]

    # One field builder for both paths, so a create and an overwrite cannot
    # drift into writing different subsets of what the seed owns.
    def _fields(**overrides: Any) -> AgentEditFields:
        # Every other field is explicitly None so the profile inherits the
        # session's model and sampling settings: a seed pinning a model would
        # silently override the operator's provider choice. Spelled out rather
        # than defaulted because ``AgentEditFields`` is validated in strict
        # mode, which is the convention every other caller here follows.
        base: dict[str, Any] = dict(
            name=None,
            # ``when_to_use`` FIRST, and the order is load-bearing. The
            # registry has one description field; a profile has two texts, and
            # this one is the ROUTING text — it is what ``search`` embeds and
            # matches against. Persisting ``description`` instead silently
            # dropped the trigger phrasings on install, so a role that was
            # discoverable as a packaged starter became undiscoverable the
            # moment an operator installed it, and search then recommended a
            # confidently wrong role rather than failing visibly ("check the UI
            # looks right" -> manager).
            description=seed.when_to_use or seed.description,
            tags=tags,
            categories=["role"],
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
        agent = registry.create_agent(_fields(name=seed.name))
    else:
        agent = existing
        # An overwrite restores the seed's ROLE FIELDS too, not just its prose.
        # This branch used to write only ``system_prompt``, which made a
        # restore half a restore: an edited ``tools:`` tag survived, so a
        # `reviewer` reset after someone widened its allowlist kept the full
        # write inventory under the packaged guidance's name. The allowlist is
        # a capability boundary rather than advice, so restoring the text
        # without it fails OPEN while reporting success.
        registry.update_agent(agent.id, _fields())
    registry.set_agent_system_prompt(agent.id, seed.instructions)
    return profile_from_agent(registry, agent), False


def seed_origin(agent: "AgentData") -> str | None:
    """The seed name a role row was installed FROM, or None if self-authored.

    ``reset`` overwrites, so "is this row a copy of a packaged seed?" has to be
    answerable from the row itself rather than guessed from the name. Guessing
    from the name is what made a self-authored ``scout`` resettable: it collides
    with a packaged starter, so a name-only check called it a diverged install
    and destroyed the operator's own work.

    Rows written before this marker existed return None and are therefore
    treated as self-authored, which is the SAFE direction: the worst outcome is
    that an operator with an old installed row is told to use ``op='update'``
    instead of getting a one-shot restore, rather than a reset eating work the
    harness never wrote.
    """

    prefix = SEED_ORIGIN_PREFIX
    for tag in agent.tags or []:
        text = str(tag).strip()
        if not text.lower().startswith(prefix):
            continue
        origin = text[len(prefix) :].strip().lower()
        if not origin:
            return None
        # The marker must name THIS row, and must name a real starter. Tags are
        # writable by the server routes, the desktop UI and agent import, none
        # of which know what this marker means, so a destructive verb keying on
        # it cannot simply trust whatever string it finds: a `seed:reviewer`
        # tag carried onto an unrelated agent would otherwise hand `reset`
        # permission to overwrite that agent with the reviewer seed. Validating
        # the marker against the row's own name keeps a cross-name or forged
        # tag inert rather than dangerous.
        if origin != str(agent.name or "").strip().lower():
            logger.warning(
                "ignoring seed provenance tag %r on agent %r: it names another role",
                text,
                agent.name,
            )
            return None
        if origin not in set(list_seeds()):
            return None
        return origin
    return None


def matches_seed_text(profile: AgentProfile, seed: AgentProfile) -> bool:
    """Whether a row's PROSE is still byte-identical to the packaged seed's.

    The unlock for rows installed before provenance was recorded. A row whose
    instructions and routing description both still match the seed exactly
    cannot be self-authored work worth protecting — adopting the seed over it
    is a no-op on every text a human would have written — so it is safe to
    treat as an install even with no marker. That is what keeps the provenance
    guard from permanently locking out every role installed by an earlier
    release, without weakening it for a row that actually holds someone's
    words.

    Deliberately NOT part of :func:`seed_divergence`: this asks "is this the
    shipped text?", which is a provenance question, while divergence asks
    "should reset do anything?". Conflating them would make a role unlockable
    by the very edit that makes it worth restoring.
    """

    return (
        seed.instructions.strip() == (profile.instructions or "").strip()
        and (seed.when_to_use or seed.description).strip()
        == (profile.description or profile.when_to_use or "").strip()
    )


def seed_divergence(profile: AgentProfile, seed: AgentProfile) -> tuple[str, ...]:
    """Which of the seed's own fields an installed role no longer matches.

    ONE definition of "diverged", because ``show`` and ``reset`` must never
    disagree about whether a role is clean. They previously each compared the
    instruction body independently, which had two consequences: a role whose
    ``tools`` allowlist had been widened but whose prose was untouched reported
    "nothing was changed" and kept the widened surface (the exact fail-open the
    restore exists to close, while announcing success), and the discrepancy was
    escapable by editing one character of prose, which flipped the same reset
    into restoring the allowlist after all.

    Only fields the seed actually WRITES are compared. ``model``, ``hosting``
    and the sampling settings are deliberately left to the operator (a seed
    pinning a model would override their provider choice), so a difference
    there is not divergence and must not be reported as one.

    ``description`` IS compared, against ``when_to_use or description`` because
    that is the value :func:`install_seed` persists — the routing text ``search``
    embeds. An earlier version excluded it, justified by the claim that
    comparing it "would flag every installed role". That claim was false, and
    the way it was false is the point: it holds only when comparing against
    ``seed.description``, which is NOT what install writes. Measured against
    all six packaged starters, the correct comparison flags zero. Leaving it
    out meant a description-only edit was invisible and unresettable, and a
    reset triggered by any other field silently rewrote the routing text with
    no echo, so a role the user could ``search`` for stopped matching
    afterwards. Verify an exclusion by RUNNING it against every real seed, not
    by reasoning about what it would do.

    Field names are returned rather than a bool so the caller can say WHICH
    fields it is about to replace: an overwrite the user cannot see coming is
    a data loss with a friendly message on it.
    """

    diverged: list[str] = []
    if seed.instructions.strip() != (profile.instructions or "").strip():
        diverged.append("instructions")
    if (seed.when_to_use or seed.description or "").strip() != (
        profile.description or profile.when_to_use or ""
    ).strip():
        diverged.append("description")
    if (profile.tools or None) != (seed.tools or None):
        diverged.append("tools")
    if (profile.effort or None) != (seed.effort or None):
        diverged.append("effort")
    if bool(profile.may_delegate) != bool(seed.may_delegate):
        diverged.append("delegate")
    return tuple(diverged)


def seed_tags(profile: AgentProfile) -> tuple[str, ...]:
    """The ``key:value`` tags encoding a profile's role fields.

    Shared by seed installation and the ``agent`` tool so both write the exact
    same encoding; two writers with two spellings is how a ``tools:`` tag would
    later stop being read.
    """

    tags: list[str] = ["role"]
    if profile.tools:
        tags.append("tools:" + ",".join(profile.tools))
    if profile.effort:
        tags.append(f"effort:{profile.effort}")
    if profile.may_delegate:
        tags.append("delegate:yes")
    return tuple(tags)


#: Anything with a ``.name``; the tool types live in ``harness.types`` and
#: importing them here would pull the tool layer into this module's graph.
_ToolT = TypeVar("_ToolT")


def filter_tools(tools: Sequence[_ToolT], profile: AgentProfile | None) -> list[_ToolT]:
    """Filter a built tool inventory down to the profile's allowlist.

    A profile naming a tool that does not exist in this session (an MCP tool
    from another machine, a renamed builtin) simply matches nothing rather than
    raising: the role still runs, with the tools it does have.
    """

    if profile is None or not profile.tools:
        return list(tools)
    allowed = set(profile.tools)
    return [tool for tool in tools if getattr(tool, "name", None) in allowed]
