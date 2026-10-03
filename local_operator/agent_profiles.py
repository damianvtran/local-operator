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

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence, TypeVar

from local_operator.action_class import PROACTIVE as PROACTIVE_CLASS
from local_operator.action_class import TAG_KEY as CLASS_TAG_KEY
from local_operator.action_class import class_from_tags, is_class_tag
from local_operator.action_class import normalize as normalize_action_class

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

#: Tag prefix recording the packaged seed's ``version:`` frontmatter as it was
#: at INSTALL time, e.g. ``seed_version:1.0.0``. Written by
#: :func:`install_seed` alongside the ``seed:`` provenance marker, and read by
#: :func:`sync_installed_seeds` to answer "did the packaged starter move since
#: this row was installed?".
#:
#: The version alone cannot answer "was the row EDITED?" — when a seed's text
#: moves, an untouched row and an edited one both differ from the packaged
#: text, so the two are indistinguishable from versions and divergence alone.
#: That is why :data:`SEED_SHA256_PREFIX` records the fingerprint of what was
#: actually installed: it is the baseline the cleanliness check compares
#: against, and without it the required default-apply behaviour on an
#: unedited update would be unreachable (every update would look "diverged").
SEED_VERSION_PREFIX = "seed_version:"

#: Tag prefix recording the sha256 fingerprint of the seed-written fields as
#: they were installed, e.g. ``seed_sha256:9f2c…`` (see
#: :func:`seed_fingerprint`). Same single writer as the other markers.
SEED_SHA256_PREFIX = "seed_sha256:"

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
    #: The display LABEL carried by the row or declared by a packaged seed's
    #: ``label:`` frontmatter. DISPLAY-ONLY, exactly as
    #: :func:`local_operator.display_labels.display_form` defines: ``name`` is
    #: the key every resolver and surface ADDRESSES by, and this field is what
    #: a listing paints. Empty means "no stored label" -- a registry row holds
    #: its derived default after the first write (``save_agent``), and an
    #: un-installed seed falls back to its own derived default too.
    label: str = ""
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
    #: The agent's action class (``reactive`` | ``proactive``). Python cannot
    #: name an attribute ``class``, hence ``action_class`` in code; the
    #: user-facing word stays "class", the frontmatter key stays ``class:`` and
    #: the tag encoding stays ``class:proactive``. Absent everywhere ⇒
    #: ``reactive``, which is today's behaviour exactly — see
    #: :mod:`local_operator.action_class` for the whole mechanism.
    action_class: str = "reactive"
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

    # ``class:`` mirrors ``delegate``'s handling, including its tolerance: a
    # seed file is hand-edited, so a bad spelling degrades to the default
    # (reactive — the safe side, because an unrecognized class must never
    # silently ENABLE proactive behaviour) rather than failing the parse.
    action_class = normalize_action_class(meta.get("class"))

    return AgentProfile(
        name=_str("name") or name,
        # A seed's canonical label (``label: UX Reviewer``): display-only, and
        # deliberately NOT part of the seed fingerprint/divergence comparison
        # -- see ``_SEED_FIELDS``.
        label=_str("label"),
        description=_str("description"),
        when_to_use=_str("when_to_use"),
        instructions=body[:MAX_INSTRUCTIONS_CHARS],
        tools=_as_tuple(meta.get("tools")),
        effort=_str("effort") or None,
        may_delegate=may_delegate,
        action_class=action_class,
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


def load_seed_version(name: str) -> str:
    """The packaged seed's own ``version:`` frontmatter, or "" when absent.

    Read through :func:`_split_frontmatter` — the same parser the loader and
    the manifest generator use — rather than a second YAML reader beside them,
    because two parsers is how the two would later disagree about the same
    bytes. Deliberately a function rather than an ``AgentProfile`` field: the
    version exists for the published manifest and for the install stamp, and
    nothing in the resolution or divergence paths reads it (see
    ``tests/unit/test_agent_seed_manifest.py``, which pins that boundary).

    Unknown name, unreadable file and a missing key all answer "" — install
    treats that as "no version recorded" rather than refusing, because the
    fingerprint marker still makes the row updateable.
    """

    key = (name or "").strip().lower()
    if not key or key not in set(list_seeds()):
        return ""
    try:
        text = (SEEDS_DIR / f"{key}.md").read_text(encoding="utf-8", errors="replace")
    except OSError:  # pragma: no cover - unreadable package file
        logger.warning("agent seed %s could not be read for its version", key)
        return ""
    meta, _body = _split_frontmatter(text)
    return str(meta.get("version") or "").strip()


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

    # The class is read back from the SAME tags ``seed_tags`` writes, so an
    # installed proactive row resolves proactive without a schema migration —
    # and an absent tag reads reactive, which is what every pre-class row is.
    return AgentProfile(
        name=agent.name,
        # The label rides the ROW, not the tags: it is not a role field (a
        # label-only edit must never read as seed divergence -- see
        # ``_SEED_FIELDS``), so there is nothing to encode.
        label=str(getattr(agent, "label", "") or ""),
        description=str(agent.description or ""),
        when_to_use=str(agent.description or ""),
        instructions=_agent_instructions(registry, agent),
        tools=tools,
        effort=effort,
        may_delegate=may_delegate,
        action_class=class_from_tags(tags),
        agent_id=agent.id,
        model=str(agent.model or ""),
        hosting=str(agent.hosting or ""),
    )


def agent_listing_rows(registry: Any | None) -> list[tuple[str, str, str, str]]:
    """``(name, label, kind-facts, summary)`` per role/specialist, roles first.

    ONE enumeration feeding the listing block, the argument picker AND the
    detached runtime's routed ``/agent``, so no two surfaces can disagree about
    which names ``/agent`` accepts or how a name reads. It lives here rather than
    on ``OperatorApp`` because a session can be hosted by a process with no TUI
    at all: the runtime used to answer a bare ``/agent`` with a ``noop`` whose
    rows only the terminal's own resolver could draw, so the phone offered the
    command in its sheet and painting the tap delivered nothing (review round 1,
    R1-2/U2). A second assembly in ``serving.py`` would have been a second source
    of truth for the same list, which is what the original split was avoiding.

    The row carries the RAW label beside the name, never a composed display
    form: each surface composes through the ONE shared rule
    (``display_labels.display_form`` / ``bounded_display_form``) for its own
    geometry, and the picker keeps ``name`` as the value it completes --
    labels are display-only, so completion must stay on the key.

    Scope is deliberate: only rows tagged as delegation roles (``is_role``) or
    explicitly authored specialists (``is_specialist``). A registry also holds
    ordinary conversational and autosave agents; offering those would be noise at
    best and a privacy leak at worst — the same boundary the ``agent`` tool's
    listing draws. ``registry`` may be ``None`` (a session that never wired one):
    the packaged seeds below still list, because ``resolve_profile`` falls through
    to them, so ``/agent reviewer`` works on a fresh machine and the listing must
    not deny a name the attach path accepts.
    """
    from local_operator.action_class import class_from_tags
    from local_operator.action_class import normalize as normalize_action_class

    rows: list[tuple[str, str, str, str]] = []
    seen: set[str] = set()
    if registry is not None and hasattr(registry, "list_agents"):
        try:
            agents = list(registry.list_agents())
        except Exception:
            agents = []
        roles: list[tuple[str, str, str, str]] = []
        specialists: list[tuple[str, str, str, str]] = []
        for agent in agents:
            try:
                if is_role(agent):
                    profile = profile_from_agent(registry, agent)
                    facts = "role"
                    # The CLASS leads the optional facts (design round 1, D1):
                    # the settings pane truncates this line to 27 cells, and a
                    # marker appended after a configured role's model/effort was
                    # the first thing cut — the surface §8.1.3 puts forward as
                    # where the class is visible could not show it on exactly the
                    # agents that carry config detail.
                    if normalize_action_class(profile.action_class) == PROACTIVE_CLASS:
                        facts += " · proactive"
                    # Model/effort are facts a user picks a hat by; the rest of
                    # the profile is what the attach applies.
                    if profile.model:
                        facts += f" · {profile.model}"
                    if profile.effort:
                        facts += f" · effort {profile.effort}"
                    summary = (profile.when_to_use or profile.description or "").strip()
                    roles.append((profile.name, profile.label, facts, summary))
                    seen.add(profile.name.lower())
                elif is_specialist(agent):
                    summary = str(agent.description or "").strip()
                    class_fact = (
                        " · proactive" if class_from_tags(agent.tags) == PROACTIVE_CLASS else ""
                    )
                    specialists.append(
                        (
                            str(agent.name),
                            str(getattr(agent, "label", "") or ""),
                            f"specialist{class_fact}",
                            summary,
                        )
                    )
                    seen.add(str(agent.name).lower())
            except Exception:
                continue
        rows.extend(sorted(roles, key=lambda row: row[0].lower()))
        rows.extend(sorted(specialists, key=lambda row: row[0].lower()))
    seeds: list[tuple[str, str, str, str]] = []
    for seed_name in list_seeds():
        if seed_name.lower() in seen:
            continue
        profile = load_seed(seed_name)
        if profile is None:
            continue
        summary = (profile.when_to_use or profile.description or "").strip()
        seed_facts = "role · packaged"
        if normalize_action_class(profile.action_class) == PROACTIVE_CLASS:
            seed_facts += " · proactive"
        seeds.append((profile.name, profile.label, seed_facts, summary))
    rows.extend(sorted(seeds, key=lambda row: row[0].lower()))
    return rows


def resolve_profile(
    name: str | None,
    *,
    registry: Any = None,
    strict_registry: bool = False,
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

    ``strict_registry`` (the launch/resume path) turns a registry READ FAILURE
    from "fall through to the packaged seed" into a raise: a seed carries no
    operator pin, so falling through there is how a pinned role could launch
    on an un-pinned model — the silent substitution the strict tier path
    refuses one layer up. A genuinely ABSENT row still resolves to the seed
    (or ``None``), strict or not; the distinction is read-vs-absent, not
    role-vs-no-role.
    """

    key = (name or "").strip()
    if not key:
        return None
    if registry is not None:
        if strict_registry:
            # ENFORCE READABILITY FIRST, with the same gate the resume-
            # admission path and the desktop profile routes use: it raises
            # ``ProfileRegistryUnavailable`` when a definition could not be
            # read, so a caller that must not silently inherit a pinned
            # role's model can tell "genuinely no role row" (the seed/None
            # fallthrough below, unchanged) apart from "the registry could
            # not be read". Duck-typed like every other access here — hosts
            # attach any object answering the two methods used below.
            complete = getattr(registry, "require_complete_metadata", None)
            if callable(complete):
                complete()
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
            if strict_registry:
                # A read that fails even after the completeness gate is still
                # not "no role of that name": surface it instead of falling
                # through to a seed, which could silently drop an operator
                # pin in exactly the case strict callers must refuse.
                raise
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
        if existing is None:
            # Fall back to the SAME fold every other resolver uses
            # (``resolve_profile``, the desktop routes, sync's own row scan).
            # The exact-case lookup misses a row the operator renamed to a
            # different spelling (``Reviewer`` over the packaged ``reviewer``),
            # and missing it here is how a second row with the packaged
            # spelling got minted beside the renamed one — including on sync's
            # apply path, which reported an update while the duplicate appeared
            # (agent review round 1, M2).
            folded = _role_named(registry, seed.name.strip().lower())
            if folded is not None:
                existing = folded
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

    # The provenance markers ride alongside the profile's own field tags. The
    # ``seed:`` marker is what later lets ``reset`` tell an installed copy from
    # a role the operator authored under a name that happens to collide with a
    # starter; the version and fingerprint are what ``sync_installed_seeds``
    # compares against to answer "did the packaged starter move since install,
    # and has this copy been edited?" — see the constants' docstrings for why
    # the version alone is not enough. ONE writer, both sync inputs: a second
    # site that stamped either marker is how the two would quietly disagree.
    tags = [
        *seed_tags(seed),
        f"{SEED_ORIGIN_PREFIX}{seed.name}",
        f"{SEED_SHA256_PREFIX}{seed_fingerprint(seed)}",
    ]
    version = load_seed_version(seed.name)
    if version:
        # A seed whose frontmatter has no ``version:`` is a broken package —
        # the manifest generator refuses to render one — but install is not the
        # place to raise a user-visible error over it: the fingerprint alone
        # still proves cleanliness, so the row stays updateable, and sync
        # reports the version as unknown rather than inventing one.
        tags.append(f"{SEED_VERSION_PREFIX}{version}")

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
            # The seed's canonical label ships on install AND reset (""
            # re-derives -- see ``AgentEditFields.label``); the reset path
            # deliberately writes it as a plain field update, NOT through the
            # divergence gate, because a label edit is not "edited since
            # install" (``_SEED_FIELDS``).
            label=seed.label or "",
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


#: The fields a packaged seed WRITES into a role, in the order divergence
#: reports them and the fingerprint hashes them. One list so the two cannot
#: drift: a field added to the comparison but not the fingerprint (or the
#: reverse) would let sync call an edited row clean or an untouched row edited.
#:
#: ``class`` rides here (rather than beside the seed-only ``version:`` stamp)
#: because a seed CAN declare it (Aida ships ``class: proactive``) and the tag
#: encoder writes it into the row — so it is a field the seed writes, the
#: exact category this list is for. A user's switch to reactive therefore
#: shows as divergence and ``reset`` restores the packaged class, the same
#: deal as ``delegate``.
#: ``label`` is deliberately NOT in this tuple, mirroring teams' local-only
#: stance: the label is display metadata a user may retitle without having
#: "edited" the seed's ROLE. Excluded from the comparison AND the fingerprint,
#: a label-only edit never shows as divergence, never blocks a sync, and a
#: ``reset`` still restores the packaged label because it rides
#: ``install_seed._fields``. One list stays one list: the field a reset writes
#: and the field divergence compares are allowed to differ ONLY where a
#: docstring says so, which is here.
_SEED_FIELDS: tuple[str, ...] = (
    "instructions",
    "description",
    "tools",
    "effort",
    "delegate",
    "class",
)


def _seed_field_values(profile: AgentProfile) -> tuple[Any, ...]:
    """The seed-written fields, canonicalized, in ``_SEED_FIELDS`` order.

    ``when_to_use or description`` on both sides, which is the seed side of the
    historical divergence comparison: a registered profile carries the SAME
    string in both fields (``profile_from_agent``), so for a row the two
    spellings are indistinguishable, while a packaged seed's routing text may
    live only in ``when_to_use``. Normalizing once here is what keeps
    :func:`seed_divergence` and :func:`seed_fingerprint` from disagreeing about
    what "the same fields" means.
    """

    return (
        (profile.instructions or "").strip(),
        (profile.when_to_use or profile.description or "").strip(),
        tuple(profile.tools) if profile.tools else None,
        profile.effort or None,
        bool(profile.may_delegate),
        normalize_action_class(profile.action_class),
    )


def seed_fingerprint(profile: AgentProfile) -> str:
    """A stable sha256 of the fields :func:`seed_divergence` compares.

    This is the baseline that makes "has this row been EDITED since install?"
    decidable. Version stamps alone cannot answer it: once the packaged text
    moves, an untouched row and an edited one both diverge from the package, so
    a sync keyed on versions and divergence would have to either overwrite
    unedited installs only-with-``force`` (making the feature useless for its
    one headline job) or overwrite edits silently. Recording what was actually
    installed settles it exactly, and covers every field the seed writes — not
    just the prose — because a widened ``tools:`` allowlist is divergence the
    restore exists to protect (see :func:`seed_divergence`).

    Deterministic by construction: ``json.dumps`` of a fixed-order list of
    primitives, non-ASCII kept literal, separators pinned.
    """

    payload = json.dumps(
        list(_seed_field_values(profile)), ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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

    mine = _seed_field_values(profile)
    packaged = _seed_field_values(seed)
    return tuple(
        field for field, value, expected in zip(_SEED_FIELDS, mine, packaged) if value != expected
    )


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
    # THE CLASS IS WRITTEN FOR EVERY PROFILE, ``reactive`` INCLUDED, and that is
    # not symmetry for its own sake: this helper is the ONE encoder of a row's
    # role fields, and the paths that rebuild a row from a profile — an ordinary
    # role edit (``tools/agent_tool.write_profile``), the desktop profile route,
    # ``install_seed`` — all come through it. While it omitted the tag for a
    # reactive profile, ANY such edit STRIPPED an explicit ``class:reactive``
    # back off the row, restoring the very ambiguity the repair for pre-class
    # rows keys on: the operator's own ``/agent class aida reactive`` stayed
    # durable only until someone fixed a typo in her prompt (agent review round
    # 1, R1 — measured on his own row, which carries a stale install
    # fingerprint and so was one edit away from a silent re-arm). Writing it
    # here means no caller can forget: an absent tag then means exactly one
    # thing, "never classified", which is the state pre-class installs are in.
    tags.append(f"{CLASS_TAG_KEY}:{normalize_action_class(profile.action_class)}")
    return tuple(tags)


#: The class a packaged starter declares, when it declares one at all. The
#: backfill's whole question is "does this starter want a class?" — a starter
#: with no ``class:`` frontmatter never does, which is what keeps the repair off
#: the ten seeds that carry none.
SEED_CLASS_PROACTIVE = PROACTIVE_CLASS


def backfill_seed_action_class(
    config_dir: Path, *, registry: "AgentRegistry | None" = None
) -> tuple[str, ...]:
    """Give installed rows the class tag their packaged starter declares.

    THE GAP THIS CLOSES, and why it needs a repair rather than a read-side
    default. ``class:`` frontmatter, the ``class:<value>`` tag and the readers
    that gate on it (Aida's cadence, patience waits, the trigger layer's own
    mirror) all arrived in ONE release, and the install path is the only writer
    of a row's tags. Every role row installed by an earlier release therefore
    carries no class tag at all, and every reader normalizes an absent tag to
    ``reactive`` — so an install that had a working cadence the day before goes
    silent on upgrade, with no user action, no message and nothing on disk to
    say why. Measured on the operator's live install on 2026-09-30: the ``aida``
    row was installed 2026-09-28 at ``seed_version:1.0.0``, the class feature
    shipped in v0.64.9 that morning, and the next session open dropped her
    cadence row (``filter_on_load``), stopped re-arming it (``reconcile``'s
    reactive branch) and left her escalation tray unconsumed.

    THE PREDICATE, in the order it is checked — every step narrows the blast
    radius:

    1. a ROLE row (only these are attachable, and only these carry a switch);
    2. seed-origin (``seed:<name>``: the row is a copy of a packaged starter);
    3. NO ``class:`` tag of its own — see :func:`action_class.with_class_tag`
       for why an absence now means "never classified";
    4. the packaged starter still exists and DECLARES a class, which today
       means ``proactive``: a starter that declares nothing is never touched,
       so a row in a reactive-by-nature starter cannot be re-classed;
    5. the row records an install fingerprint, and it DIFFERS from the packaged
       starter's — i.e. the starter has moved since this row was installed.
       This is what keeps the repair off a deliberate switch: a switch writes
       ``class:reactive`` (step 3 already excluded it), and a row whose
       recorded fingerprint IS the current starter was written by code that
       emits the tag, so a missing one there was removed by a hand edit rather
       than by an upgrade.

    Residual, stated rather than implied: a row from an OLDER starter whose
    class was switched to reactive in the hours between the class feature
    shipping and this backfill existing removed its tag (that was the old
    encoding) and would be flipped back once. It is bounded by the feature's
    own age, it is logged at INFO, and the operator's next switch — now writing
    an explicit tag — is durable.

    Idempotent: after one pass every matching row carries the tag, so a second
    call is a read-only no-op. Best-effort per row (a locked or unreadable
    registry answers ``()``), and it returns the names it flipped, which the
    startup seam logs and the tests assert on.

    IT REFUSES BEFORE CONSTRUCTING A WRITER, and that ordering is the whole of
    the guard. ``AgentRegistry.__init__`` mkdirs the config dir *and* ``agents/``
    (``agents.py``), and this runs from the ``lop`` entry point for EVERY
    subcommand — including the ones whose documented contract is that a
    storeless machine stays storeless (a bare ``config list``, ``login``,
    ``--version``, the org-sharing push, the interactive shell's pre-session
    guard). Constructing the registry to ask "is there anything to repair?"
    created the store it was asking about (agent review round 2, R5: three guard
    suites caught it, on paths that had never written a byte). A machine with no
    ``agents/`` directory has no rows by definition, so the directory test IS the
    first term of the predicate — nothing scanned, nothing created — and the
    repair still runs on every machine that has ever installed a role.
    """

    from local_operator.action_class import PROACTIVE, set_registered_action_class

    if registry is None and not (Path(config_dir) / "agents").is_dir():
        return ()

    try:
        from local_operator.agents import AgentRegistry

        reg = registry if registry is not None else AgentRegistry(config_dir)
        rows = list(reg.list_agents())
    except Exception:  # noqa: BLE001 — no registry, no rows: nothing to repair
        logger.debug("class backfill: no readable registry", exc_info=True)
        return ()

    changed: list[str] = []
    for agent in rows:
        try:
            name = str(getattr(agent, "name", "") or "")
            if not name or not is_role(agent):
                continue
            if any(is_class_tag(tag) for tag in (getattr(agent, "tags", None) or ())):
                continue
            origin = marker_value(agent, SEED_ORIGIN_PREFIX)
            if not origin:
                continue
            seed = load_seed(origin)
            if seed is None or normalize_action_class(seed.action_class) != SEED_CLASS_PROACTIVE:
                continue
            baseline = _installed_fingerprint(agent)
            if baseline is None:
                # No install record: "did the starter move?" is unanswerable,
                # and the same refusal ``_sync_one_seed`` makes applies here —
                # a row we cannot classify is left exactly as it is.
                logger.info(
                    "class backfill: %s has no install fingerprint; leaving its class alone",
                    name,
                )
                continue
            if baseline == seed_fingerprint(seed):
                continue
            set_registered_action_class(reg, name, PROACTIVE)
            changed.append(name)
        except Exception:  # noqa: BLE001 — one unreadable row never stops the pass
            logger.warning("class backfill: skipping row %r", getattr(agent, "name", None))
            logger.debug("class backfill: traceback", exc_info=True)
    if changed:
        logger.info(
            "class backfill: %s gained the class tag their packaged starters declare",
            ", ".join(sorted(changed)),
        )
    return tuple(changed)


# -- update checks -----------------------------------------------------------


@dataclass(frozen=True)
class SeedSyncVerdict:
    """The outcome of checking ONE installed seed-origin row for updates.

    One row per verdict, in the caller's requested order (or by name when
    nothing was requested), so every surface — the ``agent`` tool, the CLI and
    the desktop route — renders the same list rather than deriving its own.

    ``applied`` is the part a caller must not infer from ``verdict``: an
    ``outdated-clean`` row is written on the spot, an ``outdated-diverged`` row
    is written only when the caller passed ``force``, and a caller that
    reported "updated" for a refused apply would be this feature's version of
    every over-claiming receipt this repo has had to fix.
    """

    name: str
    verdict: Literal["up-to-date", "outdated-clean", "outdated-diverged", "not-installed"]
    installed_version: str | None = None
    packaged_version: str = ""
    diverged_fields: tuple[str, ...] = ()
    applied: bool = False
    #: The instructions REPLACED by an applied update, kept verbatim so the
    #: overwrite stays recoverable by copy-paste — the same guarantee
    #: ``op='reset'`` makes, and the reason the clean arm may overwrite at all.
    replaced_instructions: str | None = None
    #: ``(field, old value)`` for the non-prose fields an apply replaced, in
    #: :data:`_SEED_FIELDS` order: the reset echo's second half, so a forced
    #: overwrite of a widened ``tools:`` allowlist is still recoverable.
    replaced_fields: tuple[tuple[str, str], ...] = ()
    detail: str = ""


def _field_text(value: Any) -> str:
    """A replaced field value as one line — ``reset``'s own echo spelling.

    ``None``/empty answer ``(unset)`` rather than a blank, because a blank line
    beside ``your tools:`` reads as a truncation, and a bool answers yes/no so
    ``your delegate: False`` cannot be misread as a string field.
    """

    if value is None or value == "" or value == ():
        return "(unset)"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, tuple):
        return ", ".join(str(item) for item in value)
    return str(value)


#: A recorded fingerprint is exactly a sha256 hex digest, or it is not usable
#: as "what was installed" and the row is treated as unprovable rather than
#: matched against garbage. Shared by the seed and hub readers (agent review
#: round 1, n2 — the two regexes were twins).
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def is_sha256_hex(value: str) -> bool:
    """Whether ``value`` is exactly a sha256 hex digest (any case)."""

    return bool(_SHA256_HEX_RE.match(value.strip().lower()))


def marker_value(agent: "AgentData", prefix: str) -> str | None:
    """The value of the FIRST tag carrying ``prefix`` (case-insensitive), or None.

    Markers are provenance, not authority: the tags array is writable through
    the desktop routes, agent import and the tool itself, so every reader of a
    marker treats an absent or malformed value as "not recorded" rather than
    repairing it into something trusted.

    Shared with the hub-provenance readers in :mod:`local_operator.agents`:
    ``seed:`` and ``hub:`` markers resolve the same way, and two
    implementations is how one of them would later stop treating a malformed
    marker as "not recorded" (agent review round 1, n2).
    """

    for tag in agent.tags or []:
        text = str(tag).strip()
        if not text.lower().startswith(prefix):
            continue
        value = text[len(prefix) :].strip()
        return value or None
    return None


def _installed_fingerprint(agent: "AgentData") -> str | None:
    """The recorded install fingerprint, or None when missing/malformed."""

    value = marker_value(agent, SEED_SHA256_PREFIX)
    if value is None or not is_sha256_hex(value):
        return None
    return value.lower()


def _installed_seed_rows(registry: "AgentRegistry") -> dict[str, "AgentData"]:
    """Installed seed-origin ROLE rows, keyed by the seed name they fold to.

    ``is_role`` is required on top of :func:`seed_origin` for the same reason
    ``op='reset'`` checks it before overwriting: an archive can carry a
    ``seed:`` tag onto a row that is not a role, and an update would then
    rewrite that row's prompt with role guidance it never asked for. Reads
    stay tolerant (the caller lists whatever the marker names); writes are
    what need the guard.
    """

    rows: dict[str, "AgentData"] = {}
    for agent in registry.list_agents():
        if not is_role(agent):
            continue
        origin = seed_origin(agent)
        if origin is None:
            continue
        rows[str(origin).strip().lower()] = agent
    return rows


def sync_installed_seeds(
    registry: "AgentRegistry", *, names: Sequence[str] | None = None, force: bool = False
) -> list[SeedSyncVerdict]:
    """Check installed seed-origin roles against the packaged starters; update them.

    The update half of "built-in agents have a way to pull the latest": the
    package ships an improved prompt, and this is the one call that notices —
    never automatically on boot (version drift is REPORTED, not applied; the
    first-run-materialises-on-demand property of seeds is deliberate), and
    never as a second implementation per surface. The ``agent`` tool, the
    ``sync`` CLI command and the desktop route all funnel through here.

    ``names=None`` means every installed seed-origin row; a sequence restricts
    the run to those names (case-folded, the way ``load_seed`` folds) and
    yields a ``not-installed`` verdict for a name with nothing installable
    behind it.

    Classification, in order:

    * no divergence from the packaged seed → ``up-to-date`` (nothing to write);
    * the installed stamp equals the packaged version → ``up-to-date`` with a
      note when local edits are present: the starter has not moved, so sync
      has nothing to pull — restoring local edits is ``op='reset'``'s job;
    * the row still holds EXACTLY what was installed (fingerprint match) while
      the package moved → ``outdated-clean``, applied immediately through
      :func:`install_seed` with ``overwrite=True``, echoing what it replaced;
    * anything else that differs → ``outdated-diverged``, reported with the
      diverged field list and applied only with ``force=True``.

    Rows installed before the stamps existed have no fingerprint, so they
    cannot PROVE cleanliness; a moved starter therefore lands in
    ``outdated-diverged`` and needs ``--force`` once. That is the same safe
    direction :func:`seed_origin` documents for unmarked rows: withhold the
    overwrite, never guess it.
    """

    rows = _installed_seed_rows(registry)
    requested = None
    if names is not None:
        requested = []
        for raw in names:
            key = str(raw).strip().lower()
            if key and key not in requested:
                requested.append(key)

    verdicts: list[SeedSyncVerdict] = []
    targets = sorted(rows) if requested is None else requested
    for key in targets:
        agent = rows.get(key)
        if agent is None:
            verdicts.append(
                SeedSyncVerdict(
                    name=key,
                    verdict="not-installed",
                    detail=_missing_seed_detail(registry, key),
                )
            )
            continue
        verdicts.append(_sync_one_seed(registry, key, agent, force=force))
    return verdicts


def _missing_seed_detail(registry: "AgentRegistry", key: str) -> str:
    """Why a requested name produced no installed seed-origin row.

    Three different situations, and each sends the reader to a different verb:
    a name nobody installed (the fix is ``op='install'``), a name whose row
    exists but was authored rather than installed (the fix is ``op='update'``,
    and the row is NOT a copy of the starter), and a name that is not a
    starter at all (there is no fix here — the caller asked for the wrong
    thing). One collapsed sentence would routinely name the wrong one.
    """

    if _role_named(registry, key) is not None:
        return "this role was not installed from a packaged starter, so there is nothing to update"
    if load_seed(key) is not None:
        return "the packaged starter exists; install it with op='install'"
    return "no installed role of this name"


def _role_named(registry: "AgentRegistry", key: str) -> "AgentData | None":
    """Any role row whose name folds to ``key``, installed or not."""

    for agent in registry.list_agents():
        if str(agent.name or "").strip().lower() == key:
            return agent
    return None


def _sync_one_seed(
    registry: "AgentRegistry", key: str, agent: "AgentData", *, force: bool
) -> SeedSyncVerdict:
    """Classify (and, where allowed, apply) one seed-origin row."""

    seed = load_seed(key)
    if seed is None:  # pragma: no cover - seed_origin already validated the catalogue
        return SeedSyncVerdict(
            name=key,
            verdict="not-installed",
            detail="the packaged starter for this row is no longer in the catalogue",
        )

    profile = profile_from_agent(registry, agent)
    installed_version = marker_value(agent, SEED_VERSION_PREFIX)
    packaged_version = load_seed_version(key)
    diverged = seed_divergence(profile, seed)
    baseline = _installed_fingerprint(agent)
    mine = _seed_field_values(profile)

    def _verdict(**overrides: Any) -> SeedSyncVerdict:
        base: dict[str, Any] = dict(
            name=key,
            installed_version=installed_version,
            packaged_version=packaged_version,
            diverged_fields=diverged,
        )
        base.update(overrides)
        return SeedSyncVerdict(**base)

    def _replaced_fields() -> tuple[tuple[str, str], ...]:
        """``(field, old value)`` for the non-prose fields an apply replaces.

        The reset echo, kept here too: ``your tools: read, grep`` is the line
        that makes a forced overwrite recoverable, and the instructions (the
        multi-line part) ride separately in ``replaced_instructions``.
        """

        values = dict(zip(_SEED_FIELDS, mine))
        return tuple(
            (field, _field_text(values.get(field))) for field in diverged if field != "instructions"
        )

    def _apply() -> bool:
        """Overwrite the classified row with the packaged starter.

        Targets ``agent.name`` — the row THIS verdict is about — not the folded
        ``key``. Install now falls back to the same fold, but naming the row
        exactly keeps the write on the row the report describes, and it is the
        direct fix for the case-renamed row folded discovery finds while the
        exact apply lookup used to miss it (there minting a duplicate
        ``reviewer`` beside ``Reviewer``; agent review round 1, M2).
        """

        return install_seed(agent.name, registry=registry, overwrite=True) is not None

    if not diverged:
        return _verdict(
            verdict="up-to-date",
            detail=f"matches the packaged starter ({packaged_version or 'unversioned'})",
        )

    if baseline is None:
        # No install record, so "did the starter move?" is unanswerable: the
        # row differs from the packaged text, and whether that is because
        # someone edited it or because the package changed cannot be told from
        # what was recorded (nothing was). The refusal is the safe direction,
        # and the wording says exactly what is known instead of announcing an
        # update or a local edit it cannot prove (agent review round 1, M1).
        applied = _apply() if force else False
        return _verdict(
            verdict="outdated-diverged",
            applied=applied,
            replaced_instructions=(profile.instructions or "") if applied else None,
            replaced_fields=_replaced_fields() if applied else (),
            detail=(
                "forced over this copy's text"
                if applied
                else (
                    "no install record — cannot tell whether the starter moved or this "
                    "copy was edited; re-run with force to take the packaged text"
                )
            ),
        )

    if seed_fingerprint(seed) == baseline:
        # The packaged starter still holds what this row was installed from, so
        # the difference is a local edit and there is nothing to PULL — applying
        # would mean silently reverting someone's work, which is ``reset``'s
        # explicit job. Decided by the FINGERPRINT, not the version string: a
        # body change shipped without a version bump is a real package move,
        # and comparing versions first is how it was mis-reported as "local
        # edits" and became un-updateable even with force (review round 1, M1).
        return _verdict(
            verdict="up-to-date",
            detail=(
                f"no update to pull (installed {installed_version}); "
                "this copy has local edits — see op='show'"
            ),
        )

    # The packaged starter moved. Direction is deliberately not gated on version
    # ORDER: sync means "make this copy match the starter THIS build ships", so
    # a downgrade (``lop`` downgraded, channel switched, a starter reverted) is
    # the same update in the other direction — both versions and the replaced
    # text ride in the receipt, so it is never silent (QA round 1, Q-1, recorded
    # in the remediation as the intended call).
    if seed_fingerprint(profile) == baseline:
        # Provably untouched since install: apply. This is the ordinary "user
        # updates local-operator, runs sync" path, and it is safe precisely
        # because the fingerprint proves no local edit can be lost — the echo of
        # the replaced text is kept anyway, matching reset.
        applied = _apply()
        return _verdict(
            verdict="outdated-clean",
            applied=applied,
            replaced_instructions=profile.instructions or "",
            replaced_fields=_replaced_fields() if applied else (),
            detail="the packaged starter changed; this copy was unedited",
        )

    # Moved and not provably clean: never overwrite a possibly-edited prompt
    # without an explicit force — the same refusal `reset` makes for rows it
    # cannot prove the harness wrote.
    applied = _apply() if force else False
    return _verdict(
        verdict="outdated-diverged",
        applied=applied,
        replaced_instructions=(profile.instructions or "") if applied else None,
        replaced_fields=_replaced_fields() if applied else (),
        detail=(
            "forced over local edits"
            if applied
            else "re-run with force to replace it"
            + (f" (installed {installed_version})" if installed_version else "")
        ),
    )


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
