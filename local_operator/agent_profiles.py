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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence, TypeVar

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


def load_seed_class(name: str) -> str | None:
    """The packaged seed's declared ``class:`` frontmatter, or None when absent.

    Parallel to :func:`load_seed_version` and read through the same parser for
    the same reason. ``None`` means "this seed declares no class" - distinct
    from a declared ``reactive`` - and the distinction is what lets the
    ledger's tail guard and the writer's class rule tell "follows its
    revision" apart from "deliberately switched" without guessing. A missing
    or unreadable file answers None, the same safe direction as the version.
    """

    key = (name or "").strip().lower()
    if not key or key not in set(list_seeds()):
        return None
    try:
        text = (SEEDS_DIR / f"{key}.md").read_text(encoding="utf-8", errors="replace")
    except OSError:  # pragma: no cover - unreadable package file
        return None
    meta, _body = _split_frontmatter(text)
    raw = meta.get("class")
    if raw is None or not str(raw).strip():
        return None
    return normalize_action_class(raw)


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


#: The field list the ORIGINAL fingerprint formula hashed, v0.63.5-v0.64.8:
#: ``_SEED_FIELDS`` minus ``class``. Kept so a row installed before the class
#: feature can still be proven unchanged - its stamp recomputes under this
#: list even though the current formula appends ``class`` and produces a
#: different digest for the same row (the skew behind finding F1 of #2060).
_LEGACY_SEED_FIELDS: tuple[str, ...] = (
    "instructions",
    "description",
    "tools",
    "effort",
    "delegate",
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

    return _seed_field_values_for(profile, _SEED_FIELDS)


def _seed_field_values_for(profile: AgentProfile, fields: tuple[str, ...]) -> tuple[Any, ...]:
    """The values of ``fields``, canonicalised exactly as the fingerprints hash.

    ONE implementation for both eras: the legacy 5-field list and the current
    6-field one must agree field-for-field, or a stamp recomputed under the
    wrong reading would misclassify silently. An unknown name is a programmer
    error in the field tuple and raises rather than participating in a digest.
    """

    values: list[Any] = []
    for field in fields:
        if field == "instructions":
            values.append((profile.instructions or "").strip())
        elif field == "description":
            values.append((profile.when_to_use or profile.description or "").strip())
        elif field == "tools":
            values.append(tuple(profile.tools) if profile.tools else None)
        elif field == "effort":
            values.append(profile.effort or None)
        elif field == "delegate":
            values.append(bool(profile.may_delegate))
        elif field == "class":
            values.append(normalize_action_class(profile.action_class))
        else:  # pragma: no cover - a typo in a field tuple, caught by tests
            raise ValueError(f"unknown seed field {field!r}")
    return tuple(values)


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


def _legacy_fingerprint(profile: AgentProfile) -> str:
    """``seed_fingerprint`` under the ORIGINAL 5-field list (v0.63.5-v0.64.8).

    The same canonicalisation and digest, over :data:`_LEGACY_SEED_FIELDS` -
    i.e. exactly what the formula computed before commit ``0b2dc18deb``
    appended ``class`` to the field list.
    """

    payload = json.dumps(
        list(_seed_field_values_for(profile, _LEGACY_SEED_FIELDS)),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stamp_family_clean(profile: AgentProfile, stamp: str | None) -> bool:
    """Whether an install stamp recomputes from the profile under EITHER era.

    Two formulas shipped, and a row's stamp names the one that was current
    when it was written: the original 5-field digest and the current 6-field
    one that appends ``class``. Accepting both is what keeps a v0.63.5-v0.64.8
    install provably unchanged today - the current-formula-only comparison
    marked every such row "diverged", the skew finding F1 of #2060 describes.
    A malformed stamp is never clean; the same refusal ``_installed_fingerprint``
    documents for unusable values.
    """

    if not stamp or not is_sha256_hex(stamp):
        return False
    return _legacy_fingerprint(profile) == stamp or seed_fingerprint(profile) == stamp


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


# -- revision ledger ---------------------------------------------------------
#
# WHY A LEDGER, when ``seed_sha256:`` already records what was installed. Two
# measured failures make the recorded stamp insufficient on its own:
#
#   * FORMULA SKEW. ``seed_fingerprint`` hashes a fixed field list, and commit
#     ``0b2dc18deb`` (shipped in v0.64.9) appended ``class`` to it. Every stamp
#     written by v0.63.5-v0.64.8 recomputes to a DIFFERENT value under the
#     current formula, so a row those releases installed can never again be
#     proven clean however untouched it is - and a plain ``sync`` answered
#     "re-run with force". The reporter's aida was exactly this row.
#   * DIRECTION. Version strings repeat (aida shipped four ``1.0.0`` builds,
#     two of them same-day), so comparing a row against the packaged text
#     cannot say whether a difference means the row is BEHIND the package or
#     AHEAD of it - and "ahead" must never be downgraded: an older ``lop``
#     starting after a newer one updated a row would flip it back, and the
#     two would ping-pong.
#
# The ledger answers both without new row state: it lists every text that
# SHIPPED in this repository's history, oldest first per seed, so a row whose
# canonical vector equals an entry is provably an unedited published revision,
# and the entry's position is what "behind" and "ahead" are defined against.
# It is generated and committed (``scripts/gen_agent_seed_revisions.py``
# re-renders it byte-identically), ships INSIDE the package
# (``agent_seeds/*.json`` package data), and - being per-COMMIT rather than
# per-release - also covers texts that reached users from ``main`` between
# releases.


#: Name of the shipped revision ledger, beside the seeds it describes. Sorted
#: seed names, oldest entry first per seed; see :func:`render_seed_revisions`
#: for the exact bytes.
SEED_REVISIONS_NAME = "seed_revisions.json"

#: Bumped when the ledger's own shape changes. One consumer refuses any other
#: value rather than guessing at a schema it does not know.
SEED_REVISIONS_SCHEMA_VERSION = 1


def normalize_seed_prose(text: str) -> str:
    """A seed's prose under the ONE canonicalisation the ledger shares.

    CRLF/CR -> LF, per-line trailing whitespace stripped, outer blank lines
    stripped. Measured the IDENTITY on all 81 HEAD-reachable committed seed
    versions (every whitespace it would move is already absent), so this is
    cheap insurance rather than a repair: the ledger compares texts written by
    many hands over months, and a careless future edit must not turn an
    untouched row into a false "diverged".

    NEVER applied to ``tools:``/``effort``/``delegate``/``class`` - those are
    capability boundaries, and an allowlist entry is a name, not prose. Only
    the two text fields are ever passed through here.
    """

    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def seed_revision_vector(profile: AgentProfile) -> tuple[Any, ...]:
    """The fields that IDENTIFY a published revision, canonicalised.

    Five fields - instructions, routing text, tools, effort, delegate - and
    ``class`` is deliberately NOT one of them. The exclusion is what makes the
    reporter's own row matchable: aida's row was migrated by the class
    backfill to ``class:proactive`` while the ledger's v1.0.0 entry records no
    class (it shipped before the feature), so a six-field identity would leave
    the exact row this feature exists for unmatchable. ``class`` is still
    CARRIED per entry - it is what the narrow writer's class rule reads - it
    simply does not participate in identity. Instructions ride as the
    whitespace-normalised string; the sha256 is taken over exactly that string.
    """

    return (
        normalize_seed_prose(profile.instructions or ""),
        normalize_seed_prose(profile.when_to_use or profile.description or ""),
        tuple(profile.tools) if profile.tools else None,
        profile.effort or None,
        bool(profile.may_delegate),
    )


@dataclass(frozen=True)
class SeedRevision:
    """One published revision of one seed: what it said, and where it landed.

    ``sha`` is the 40-hex commit that introduced the text in this repository's
    history; ``version`` is the seed's own frontmatter version AT THAT COMMIT,
    deliberately allowed to repeat (aida shipped four ``1.0.0`` builds) - it
    is a label, not an identity. Identity is :func:`seed_revision_vector`'s
    five fields; ``action_class`` is carried but excluded from identity (see
    that function) and is what the narrow writer's class rule consults. ``None``
    there means "this revision declares no class at all", which is NOT the
    same as a declared ``reactive``.
    """

    sha: str
    version: str
    instructions_sha256: str
    description: str
    tools: tuple[str, ...] | None
    effort: str | None
    delegate: bool
    action_class: str | None


def _seed_revision_from_profile(
    profile: AgentProfile, *, sha: str, version: str, action_class: str | None
) -> SeedRevision:
    """Build one ledger entry from a loaded profile.

    The ONE constructor shared by the generator (which fills ``sha``/
    ``version``/``action_class`` from git history) and by tests that publish a
    scratch revision - so a hand-published entry can never spell a field
    differently from a generated one.
    """

    vector = seed_revision_vector(profile)
    return SeedRevision(
        sha=sha,
        version=version,
        instructions_sha256=hashlib.sha256(vector[0].encode("utf-8")).hexdigest(),
        description=vector[1],
        tools=vector[2],
        effort=vector[3],
        delegate=vector[4],
        action_class=action_class,
    )


def _seed_revision_payload(entry: SeedRevision) -> dict[str, Any]:
    """One entry's JSON shape, key order fixed so bytes are stable."""

    return {
        "sha": entry.sha,
        "version": entry.version,
        "instructions_sha256": entry.instructions_sha256,
        "description": entry.description,
        "tools": list(entry.tools) if entry.tools is not None else None,
        "effort": entry.effort,
        "delegate": entry.delegate,
        "class": entry.action_class,
    }


def render_seed_revisions(seeds: Mapping[str, Sequence[SeedRevision]]) -> str:
    """The exact bytes ``seed_revisions.json`` is committed with.

    ONE serialiser for the generator that writes the packaged file and for
    tests that publish a scratch revision into a scratch copy of it, because
    two spellings is how a hand-appended entry would drift from a generated
    one. Deterministic by construction: seed names sorted, entry keys in a
    fixed order, ``indent=2``, a trailing newline, and NO timestamps or
    generator versions - byte-stable across releases, so regeneration is a
    no-op for anyone who did not move a seed (the manifest's provenance trick
    is deliberately not needed here).
    """

    payload = {
        "schema_version": SEED_REVISIONS_SCHEMA_VERSION,
        "seeds": {
            name: [_seed_revision_payload(entry) for entry in seeds[name]] for name in sorted(seeds)
        },
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def make_seed_revision(
    profile: AgentProfile,
    *,
    sha: str,
    version: str,
    declared_class: str | None,
) -> SeedRevision:
    """Build one ledger entry from a profile as read at one commit.

    ``declared_class`` is the frontmatter's declared ``class:`` (already
    normalised) or None when the file declares none - NOT
    ``profile.action_class``, which defaults to ``reactive`` for every seed
    and would erase the "no class declared" distinction the writer's class
    rule depends on (aida's row is the case: her v1.0.0 entry must record no
    class). ONE constructor, used by the generator and by tests' scratch
    ``publish_seed`` helpers, so a hand-appended entry cannot drift from a
    generated one.
    """

    vector = seed_revision_vector(profile)
    return SeedRevision(
        sha=sha,
        version=version or "",
        instructions_sha256=hashlib.sha256(vector[0].encode("utf-8")).hexdigest(),
        description=vector[1],
        tools=vector[2],
        effort=vector[3],
        delegate=vector[4],
        action_class=(
            normalize_action_class(declared_class) if declared_class is not None else None
        ),
    )


def _parse_seed_revision(row: object) -> SeedRevision | None:
    """One JSON row as a ``SeedRevision``, or None when malformed.

    Tolerant on read, strict in effect: a malformed entry is DROPPED rather
    than repaired, because a repaired entry would be a claim about a published
    text nobody can reproduce. A dropped entry costs at worst "no ledger proof"
    for the rows it described, which is the same safe direction every marker
    reader in this module takes.
    """

    if not isinstance(row, dict):
        return None
    sha = str(row.get("sha") or "").strip()
    instructions_sha256 = str(row.get("instructions_sha256") or "").strip()
    description = row.get("description")
    tools = row.get("tools", None)
    effort = row.get("effort", None)
    delegate = row.get("delegate", None)
    action_class = row.get("class", None)
    version = str(row.get("version") or "")
    if not sha or not is_sha256_hex(instructions_sha256):
        return None
    if not isinstance(description, str):
        return None
    if tools is not None and not (
        isinstance(tools, list) and all(isinstance(item, str) for item in tools)
    ):
        return None
    if effort is not None and not isinstance(effort, str):
        return None
    if not isinstance(delegate, bool):
        return None
    if action_class is not None and not isinstance(action_class, str):
        return None
    return SeedRevision(
        sha=sha,
        version=version,
        instructions_sha256=instructions_sha256.lower(),
        description=description,
        tools=tuple(tools) if tools is not None else None,
        effort=effort or None,
        delegate=delegate,
        action_class=(normalize_action_class(action_class) if action_class is not None else None),
    )


def load_seed_revisions(
    seeds_dir: Path | None = None,
) -> dict[str, tuple[SeedRevision, ...]]:
    """The packaged revision ledger, keyed by seed name. Never raises.

    A missing or corrupt file answers ``{}`` - the ledger is PROOF, not a
    precondition: without it, typed sync degrades to the pre-ledger stamp
    semantics and the startup pass stays silent, because its only proof of
    direction would be gone. That is the same safe direction ``seed_origin``
    documents for unmarked rows: withhold the overwrite, never guess it.
    """

    directory = Path(seeds_dir) if seeds_dir is not None else SEEDS_DIR
    try:
        payload = json.loads((directory / SEED_REVISIONS_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    rows = payload.get("seeds")
    if not isinstance(rows, dict):
        return {}
    revisions: dict[str, tuple[SeedRevision, ...]] = {}
    for name, entries in rows.items():
        if not isinstance(entries, list):
            continue
        parsed = tuple(
            entry for entry in (_parse_seed_revision(row) for row in entries) if entry is not None
        )
        if parsed:
            revisions[str(name).strip().lower()] = parsed
    return revisions


def _entry_matches_profile(entry: SeedRevision, profile: AgentProfile) -> bool:
    """Whether a ledger entry's identity equals the profile's canonical vector."""

    vector = seed_revision_vector(profile)
    return (
        hashlib.sha256(vector[0].encode("utf-8")).hexdigest() == entry.instructions_sha256
        and vector[1] == entry.description
        and vector[2] == entry.tools
        and vector[3] == entry.effort
        and vector[4] == entry.delegate
    )


def _match_in_entries(
    profile: AgentProfile, entries: tuple[SeedRevision, ...]
) -> tuple[int, SeedRevision] | None:
    """The last ledger entry whose identity equals the profile's vector.

    The LAST match wins: an entry repeats an earlier entry's identity only
    when the seed's own ``class`` changed without its text (the generator's
    append rule keys on the full vector), and the later entry is the one whose
    declared class the writer rule should read.
    """

    matched: tuple[int, SeedRevision] | None = None
    for index, entry in enumerate(entries):
        if _entry_matches_profile(entry, profile):
            matched = (index, entry)
    return matched


def match_seed_revision(
    name: str,
    profile: AgentProfile,
    *,
    revisions: Mapping[str, tuple[SeedRevision, ...]] | None = None,
) -> tuple[int, SeedRevision] | None:
    """The ledger position a profile's canonical vector holds, or None.

    The LAST matching index wins. An entry can repeat an earlier entry's
    identity only when the seed's own ``class`` changed without its text (the
    generator's append rule keys on the full vector), and the later entry is
    the one whose declared class the writer rule should read. Callers that
    need a position use the index; callers that need the entry's data use the
    entry.
    """

    entries = (revisions if revisions is not None else load_seed_revisions()).get(
        str(name).strip().lower()
    )
    if not entries:
        return None
    return _match_in_entries(profile, entries)


# -- update checks -----------------------------------------------------------


@dataclass(frozen=True)
class SeedSyncVerdict:
    """The outcome of checking ONE installed seed-origin row for updates.

    One row per verdict, in the caller's requested order (or by name when
    nothing was requested), so every surface — the ``agent`` tool, the CLI and
    the desktop route — renders the same list rather than deriving its own.

    ``applied`` is the part a caller must not infer from ``verdict``: an
    ``outdated-clean`` row is written on the spot WHEN the caller allowed
    writes (``apply=True``) — under ``apply=False`` it is an update AVAILABLE
    and ``applied`` stays False — while an ``outdated-diverged`` row is written
    only when the caller passed ``force``. A caller that reported "updated" for
    a refused or read-only classification would be this feature's version of
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
    #: The label a FORCED replace discarded, present only when it differed from
    #: the packaged label. The wholesale writer resets the label, and this echo
    #: is the only record of it (UX round 1, U5).
    replaced_label: str | None = None
    #: The row's non-seed tag NAMES a forced replace discarded, in row order.
    #: The wholesale writer drops every tag it does not own, so this is the
    #: only record of them (UX round 1, U5).
    replaced_tags: tuple[str, ...] = ()
    #: How many published revisions behind its packaged starter this row is,
    #: when the revision ledger could position BOTH ends of the move — a
    #: ledger-ORDER distance, never a version comparison (version strings
    #: repeat: aida shipped four ``1.0.0`` builds). ``None`` means the ledger
    #: could not position the row (the stamp-fallback semantics) or the
    #: distance does not apply (up-to-date/ahead/diverged). The startup pass's
    #: auto-apply eligibility rides on this being non-None: only a row both
    #: ends of which the ledger can place may be written unattended.
    behind_by: int | None = None
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
    registry: "AgentRegistry",
    *,
    names: Sequence[str] | None = None,
    force: bool = False,
    apply: bool = True,
) -> list[SeedSyncVerdict]:
    """Check installed seed-origin roles against the packaged starters; update them.

    The update half of "built-in agents have a way to pull the latest": the
    package ships an improved prompt, and this is the call every explicit
    surface funnels through — the ``agent`` tool, the ``sync`` CLI command and
    the desktop route. Since #2060 it is no longer the only thing that
    NOTICES: the startup seam classifies every launch
    (``startup_seed_update_pass``) and auto-applies the rows the ledger proves
    are unedited and strictly behind. This classification is single-sourced
    for both, and every apply still funnels through this call's clean arm.

    ``names=None`` means every installed seed-origin row; a sequence restricts
    the run to those names (case-folded, the way ``load_seed`` folds) and
    yields a ``not-installed`` verdict for a name with nothing installable
    behind it.

    ``apply=False`` writes no agent rows and no seed state: the starter arm
    persists nothing, and the classification is returned as verdicts. It is
    NOT a blanket byte-level promise for a whole command run — a caller that
    wraps this call in larger machinery (``lop agents sync``) may still touch
    its OWN stores, and both exceptions are pre-existing and documented: the
    hub arm refreshes ``hub/status.json``, and the startup seam's class
    backfill may repair a row's ``class:`` tag before this call runs (agent
    review round 1, R1-6; QA O1). The clean arm answers ``outdated-clean``
    with ``applied=False`` (and its ``behind_by`` distance when the ledger
    could position it), and the renderer says "update available" — a caller
    that only wants to KNOW must never be told "updated". ``lop agents sync
    --check`` and ``--dry-run`` and the startup pass's classification all use
    this mode.

    Classification, in order — the ledger (``seed_revisions.json``: every text
    that ever shipped, oldest first per seed) supplies positions, and the
    recorded install fingerprint remains the belt for rows and packages the
    ledger cannot see:

    * no divergence from the packaged seed → ``up-to-date`` (nothing to
      write);
    * the row's canonical vector equals a ledger entry AND the packaged text
      has one too → position decides: same → ``up-to-date``; the row strictly
      BEHIND → ``outdated-clean`` with ``behind_by`` set, applied through the
      narrow writer; the row AHEAD → ``up-to-date`` with "newer than this
      build ships" — never a downgrade, because an older ``lop`` must not
      flip a newer build's update back;
    * no ledger position, but the stamp recomputes from the row under either
      shipped fingerprint formula → ``outdated-clean`` (typed surfaces only:
      the startup pass refuses to auto-apply without a position — a stamp
      proves "unchanged since install", not which way the package can move);
    * the stamp equals the PACKAGED starter's fingerprint while the row
      differs → ``up-to-date`` with a local-edits note: nothing to PULL, and
      restoring local edits is ``op='reset'``'s job;
    * anything else → ``outdated-diverged``, reported with the diverged field
      list and applied only with ``force=True``.

    Rows installed before the stamps existed have no fingerprint, so they
    cannot prove cleanliness by themselves; the LEDGER is what re-proves them:
    a row whose text is a published revision is clean by construction, and the
    v0.63.5-v0.64.8 formula skew (finding F1 of #2060) is exactly that case —
    the ledger match, not the stamp, is the proof. A row the ledger cannot
    place either lands in ``outdated-diverged`` and needs the explicit replace
    once.
    """

    rows = _installed_seed_rows(registry)
    requested = None
    if names is not None:
        requested = []
        for raw in names:
            key = str(raw).strip().lower()
            if key and key not in requested:
                requested.append(key)

    revisions = load_seed_revisions()
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
        verdicts.append(
            _sync_one_seed(registry, key, agent, force=force, apply=apply, revisions=revisions)
        )
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


def _resolve_row_revision(
    name: str,
    profile: AgentProfile,
    stamp: str | None,
    revisions: Mapping[str, tuple[SeedRevision, ...]],
) -> tuple[int | None, str]:
    """``(ledger position, proof)`` for one row's current text.

    ``proof`` is ``"ledger"`` when the row's canonical vector equals a
    published revision — the position is then real, and it is what decides
    behind/ahead; ``"stamp"`` when only the recorded install fingerprint
    recomputes from the row's own fields, under EITHER shipped formula (the
    legacy 5-field one is kept so v0.63.5-v0.64.8 installs stay classifiable);
    ``"none"`` when neither holds. Only ``"ledger"`` may drive an unattended
    write: a stamp proves "unchanged since install", never which way the
    package can move.
    """

    matched = match_seed_revision(name, profile, revisions=revisions)
    if matched is not None:
        return matched[0], "ledger"
    if _stamp_family_clean(profile, stamp):
        return None, "stamp"
    return None, "none"


def _packaged_revision_index(seed: AgentProfile, entries: tuple[SeedRevision, ...]) -> int | None:
    """The ledger position of the PACKAGED starter's text, or None.

    Identity match (the five-field vector), LAST index wins — the same rule
    :func:`match_seed_revision` applies to rows, so a row and its package can
    never disagree about which entry they are. None means the packaged text
    was never published: a draft in a worktree, or a ledger that cannot see
    it; callers must then fall back to the stamp semantics, and the startup
    pass must NEVER auto-apply.
    """

    matched = _match_in_entries(seed, entries)
    return matched[0] if matched is not None else None


def _packaged_tail_entry(
    name: str, revisions: Mapping[str, tuple[SeedRevision, ...]]
) -> SeedRevision | None:
    """The ledger entry the PACKAGED starter must equal to be auto-appliable.

    The runtime guard behind "draft prompts in worktrees never auto-apply":
    the generator appends the packaged text on every release, so a properly
    generated ledger's tail IS the packaged revision - full vector, class
    included - and anything else means the text on disk was never published
    (an edit in a worktree, a rollback). Returns the tail entry when it
    matches, else None; callers must then withhold the write and stay silent.
    """

    key = str(name).strip().lower()
    entries = revisions.get(key, ())
    if not entries:
        return None
    seed = load_seed(key)
    if seed is None:
        return None
    tail = entries[-1]
    if not _entry_matches_profile(tail, seed):
        return None
    if (tail.action_class or "") != (load_seed_class(key) or ""):
        return None
    return tail


def _is_seed_owned_tag(tag: object) -> bool:
    """Whether one row tag is seed-owned (rewritten by a clean update).

    Self-describing ``key:value`` pairs whose key is one the seed encodes
    (``tools:``, ``effort:``, ``delegate:``, ``class:``, ``seed:``,
    ``seed_version:``, ``seed_sha256:``), plus the bare ``role`` category.
    Everything else — a user's ``favourite``, a foreign ``team:`` tag — is the
    row owner's and survives the narrow writer untouched. The key is
    case-insensitive, the way every tag reader here is.
    """

    key, sep, _value = str(tag).partition(":")
    normalized = key.strip().lower()
    if not sep:
        return normalized == ROLE_TAG
    return normalized in (
        "tools",
        "effort",
        "delegate",
        "class",
        "seed",
        "seed_version",
        "seed_sha256",
    )


def _resolved_seed_class(
    agent: "AgentData",
    profile: AgentProfile,
    seed: AgentProfile,
    revisions: Mapping[str, tuple[SeedRevision, ...]],
) -> str:
    """Which ``class:`` value the narrow writer stamps on a clean row.

    The rule, written down because it is subtle: the PACKAGED class is written
    iff the row has no class tag OR its class equals the class of ANY ledger
    entry whose other five fields match the row's; otherwise the row's own
    class is preserved. That is what lets aida auto-apply her prose updates —
    her row carries ``class:proactive`` (the startup backfill's repair) while
    the ledger's v1.0.0 entry records no class at all, so "preserve" is the
    right arm — while a role that merely FOLLOWED its starter's own class
    change still follows it: the row's class equals the matching entry's
    declared class, so the packaged value flows through. A deliberately
    SWITCHED row is user data like the label and is never overridden
    unattended; the packaged class reaching it is deferred to ``reset``
    (accepted trade, ADR Q2).
    """

    from local_operator.action_class import is_class_tag

    packaged_class = normalize_action_class(seed.action_class)
    if not any(is_class_tag(tag) for tag in (agent.tags or ())):
        return packaged_class
    row_class = normalize_action_class(profile.action_class)
    for entry in revisions.get(seed.name, ()):
        if _entry_matches_profile(entry, profile) and entry.action_class == row_class:
            return packaged_class
    return row_class


def _apply_seed_update(
    registry: "AgentRegistry",
    agent: "AgentData",
    seed: AgentProfile,
) -> bool:
    """Replace ONLY the seed-owned fields of a clean row, atomically.

    WHY THIS EXISTS, and why it is not ``install_seed``: the wholesale
    overwrite install/reset share rewrites the LABEL and the whole tag list,
    so a row provably untouched in the seed's fields can still lose a
    user-chosen label and a tag the user added — measured on the reporter's
    own shape (finding F4 of #2060), and unacceptable for a write the user
    never asked for. This writer touches exactly what the seed owns: the
    instructions, the routing description, the seed-owned capability tags,
    the class (under :func:`_resolved_seed_class`), and the provenance stamps.
    Label, model, categories, security prompt, sampling and every non-seed tag
    are left alone.

    ORDER IS THE CRASH CONTRACT, not style: the prompt lands FIRST (already
    atomic in the registry), the tag/stamp rewrite LAST — and the two are
    wrapped so a FAILURE of the second half ROLLS THE PROMPT BACK, leaving the
    row exactly as it was: it still classifies clean-and-behind, so the update
    simply re-applies next launch. The rollback covers the CATCHABLE case only.
    A process killed between the two writes cannot be rolled back, and the
    residual is real and narrow: the row keeps the NEW prompt with the OLD
    tags and stamps. For a prose-only revision that reads "matches the
    packaged starter" (correct text, stale stamps — cosmetic); a revision that
    ALSO moved the description leaves a row that differs in description alone
    and reads as needing an explicit ``--replace --yes``/reset — the one case
    where the crash is user-visible (agent review round 1, R1-7; UX N1).
    ``update_agent`` writes ``agent.yml`` atomically for the same class of
    reason: the strict registry gate refuses every profile while that file is
    short, and several ``lop`` processes start at once on a machine.
    """

    from local_operator.agents import AgentEditFields

    profile = profile_from_agent(registry, agent)
    class_value = _resolved_seed_class(agent, profile, seed, load_seed_revisions())
    target = AgentProfile(
        name=seed.name,
        instructions=seed.instructions,
        description=seed.description,
        when_to_use=seed.when_to_use,
        tools=seed.tools,
        effort=seed.effort,
        may_delegate=seed.may_delegate,
        action_class=class_value,
    )
    kept = [str(tag) for tag in (agent.tags or []) if not _is_seed_owned_tag(tag)]
    tags = [
        *(tag for tag in seed_tags(seed) if not tag.lower().startswith("class:")),
        *kept,
        f"{CLASS_TAG_KEY}:{class_value}",
        f"{SEED_ORIGIN_PREFIX}{seed.name}",
        f"{SEED_SHA256_PREFIX}{seed_fingerprint(target)}",
    ]
    version = load_seed_version(seed.name)
    if version:
        tags.append(f"{SEED_VERSION_PREFIX}{version}")

    description = seed.when_to_use or seed.description or ""
    previous_prompt = registry.get_agent_system_prompt(agent.id)
    registry.set_agent_system_prompt(agent.id, seed.instructions)
    # Every other field is explicitly None: ``AgentEditFields`` is validated in
    # strict mode and pyright's pydantic pass requires the full spelling - the
    # same convention ``install_seed``'s ``_fields`` documents and follows.
    try:
        registry.update_agent(
            agent.id,
            AgentEditFields(
                name=None,
                label=None,
                security_prompt=None,
                hosting=None,
                model=None,
                description=description,
                tags=tags,
                categories=None,
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
            ),
        )
    except Exception:
        # The half that landed (the prompt) is rolled back so a caught failure
        # leaves the row exactly as before and the next pass re-applies it.
        # Without this, a failure left a row that read "up-to-date" with stale
        # stamps and was never re-applied (agent review round 1, R1-7). The
        # rollback is best-effort and must not mask the original error.
        try:
            registry.set_agent_system_prompt(agent.id, previous_prompt)
        except Exception:  # noqa: BLE001 - the re-raise below is the report
            logger.warning(
                "seed update: could not roll back %r after a failed write", agent.id, exc_info=True
            )
        raise
    return True


def _sync_one_seed(
    registry: "AgentRegistry",
    key: str,
    agent: "AgentData",
    *,
    force: bool,
    apply: bool = True,
    revisions: Mapping[str, tuple[SeedRevision, ...]] | None = None,
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
    ledger = revisions if revisions is not None else load_seed_revisions()

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

    def _forced_removed() -> tuple[str | None, tuple[str, ...]]:
        """``(label, tag names)`` this row will lose to a wholesale overwrite.

        The forced path uses the wholesale writer, so it resets the label and
        drops every tag the seed does not own; both are echoed so the discard
        stays recoverable (UX round 1, U5). The label is reported only when it
        really differs from the packaged spelling - a row already showing the
        packaged label has nothing to recover.
        """

        from local_operator.display_labels import default_label, normalize_label

        label = str(getattr(agent, "label", "") or "")
        packaged_label = normalize_label(seed.label or "") or default_label(seed.name)
        removed_label = (
            label if label.strip() and label.casefold() != packaged_label.casefold() else None
        )
        removed_tags = tuple(str(tag) for tag in (agent.tags or []) if not _is_seed_owned_tag(tag))
        return removed_label, removed_tags

    def _apply_forced() -> bool:
        """Overwrite the classified row with the packaged starter, wholesale.

        Targets ``agent.name`` — the row THIS verdict is about — not the folded
        ``key``. Install now falls back to the same fold, but naming the row
        exactly keeps the write on the row the report describes, and it is the
        direct fix for the case-renamed row folded discovery finds while the
        exact apply lookup used to miss it (there minting a duplicate
        ``reviewer`` beside ``Reviewer``; agent review round 1, M2).

        The WHOLESALE writer, reserved for the explicit path (``force=True``):
        it restores the packaged label and drops non-seed tags, which is what
        "reset to the packaged version" promises — and why the unattended
        clean arm must NOT use it (finding F4 of #2060: a clean apply through
        install still reset a chosen label and lost a user's own tags).
        """

        return install_seed(agent.name, registry=registry, overwrite=True) is not None

    def _apply_clean() -> bool:
        """The unattended-safe writer: only the fields the seed owns.

        See :func:`_apply_seed_update`. The division of labour is the point:
        forced replaces keep ``install_seed``'s reset semantics, while an
        update nobody asked for must leave everything the seed does not own
        (label, non-seed tags, model, categories) exactly as it found it.
        """

        return _apply_seed_update(registry, agent, seed)

    def _clean_verdict(behind_by: int | None) -> SeedSyncVerdict:
        """The shared shape of both clean arms (ledger-positioned and stamp)."""

        applied = _apply_clean() if apply else False
        return _verdict(
            verdict="outdated-clean",
            applied=applied,
            behind_by=behind_by,
            replaced_instructions=(profile.instructions or "") if applied else None,
            replaced_fields=_replaced_fields() if applied else (),
            detail=(
                "the packaged starter changed; this copy was unedited"
                if applied
                else f"update available ({installed_version or 'installed'} -> "
                f"{packaged_version or 'packaged'})"
            ),
        )

    if not diverged:
        return _verdict(
            verdict="up-to-date",
            detail=f"matches the packaged starter ({packaged_version or 'unversioned'})",
        )

    matched_index, _proof = _resolve_row_revision(key, profile, baseline, ledger)
    packaged_index = _packaged_revision_index(seed, ledger.get(key, ()))

    # Position-decided cases first: both ends of the move are published, so the
    # ledger can say which way "different" points — the direction version
    # strings cannot (aida shipped four ``1.0.0`` builds, two of them same-day).
    if matched_index is not None and packaged_index is not None:
        if matched_index == packaged_index:
            # The same revision; only the class can differ here (identity
            # covers every other field), and a switched class is the user's to
            # keep — nothing to write, nothing to warn about.
            return _verdict(
                verdict="up-to-date",
                detail=(
                    f"no update to pull (installed {installed_version or 'unversioned'}); "
                    "this copy has local edits — leaving it alone"
                ),
            )
        if matched_index > packaged_index:
            # AHEAD: this copy holds a NEWER published revision than the
            # package being run (a downgraded ``lop``, a channel switch).
            # Silence and no write — an older build must never flip a newer
            # build's update back, or the two ping-pong the row (finding F6
            # of #2060). The report explains itself for a typed sync.
            return _verdict(
                verdict="up-to-date",
                detail=(
                    f"newer than this build ships (installed "
                    f"{installed_version or 'unversioned'}, packaged "
                    f"{packaged_version or 'unversioned'}) — leaving it alone"
                ),
            )
        # Strictly behind, provably an unedited published revision: the clean
        # arm. Applied through the narrow writer; READ-ONLY callers get the
        # same classification with applied=False so the renderer can say
        # "update available" without claiming a write that did not happen.
        return _clean_verdict(packaged_index - matched_index)

    if baseline is None and matched_index is None:
        # No install record AND no ledger position: "did the starter move?" is
        # unanswerable — the row differs from the packaged text, and whether
        # that is because someone edited it or because the package changed
        # cannot be told from anything recorded. The refusal is the safe
        # direction, and the wording says exactly what is known instead of
        # announcing an update or a local edit it cannot prove (agent review
        # round 1, M1).
        removed_label, removed_tags = _forced_removed()
        applied = _apply_forced() if (force and apply) else False
        return _verdict(
            verdict="outdated-diverged",
            applied=applied,
            replaced_instructions=(profile.instructions or "") if applied else None,
            replaced_fields=_replaced_fields() if applied else (),
            replaced_label=removed_label if applied else None,
            replaced_tags=removed_tags if applied else (),
            detail=(
                "forced over this copy's text"
                if applied
                else (
                    f"differs from the packaged starter in {', '.join(diverged)}, and there is "
                    "no install record, so it cannot be told whether the starter moved or the "
                    "copy was edited. Left alone. To take the packaged text instead: "
                    f"`lop agents sync --name {key} --replace --yes`."
                )
            ),
        )

    if seed_fingerprint(seed) == baseline or _legacy_fingerprint(seed) == baseline:
        # The packaged starter still holds what this row was installed from
        # (under either era of the formula), so the difference is a local edit
        # and there is nothing to PULL — applying would mean silently reverting
        # someone's work, which is ``reset``'s explicit job. Decided by the
        # FINGERPRINT, not the version string: a body change shipped without a
        # version bump is a real package move, and comparing versions first is
        # how it was mis-reported as "local edits" and became un-updateable
        # even with force (review round 1, M1).
        return _verdict(
            verdict="up-to-date",
            detail=(
                f"no update to pull (installed {installed_version or 'unversioned'}); "
                "this copy has local edits — leaving it alone"
            ),
        )

    if _stamp_family_clean(profile, baseline):
        # No ledger position for the packaged text (a dev worktree, a rollback)
        # or for the row, but the stamp recomputes from the row's own fields:
        # it is what SOME build installed, so typed sync keeps its pre-ledger
        # semantics — "make this copy match the starter THIS build ships", a
        # downgrade being the same update in the other direction (QA round 1,
        # Q-1) — while the startup pass refuses this path unattended: a stamp
        # proves "unchanged", never direction (finding F6 of #2060).
        return _clean_verdict(None)

    # Moved and not provably clean: never overwrite a possibly-edited prompt
    # without an explicit replace — the same refusal ``reset`` makes for rows
    # it cannot prove the harness wrote. The remedy names the flag pair the
    # CLI actually honours (``--replace --yes``); ``force`` is deprecated and
    # hidden, and prescribing it here would have sent users to a flag whose
    # own warning points at ``--replace`` (finding F2 of #2060).
    removed_label, removed_tags = _forced_removed()
    applied = _apply_forced() if (force and apply) else False
    return _verdict(
        verdict="outdated-diverged",
        applied=applied,
        replaced_instructions=(profile.instructions or "") if applied else None,
        replaced_fields=_replaced_fields() if applied else (),
        replaced_label=removed_label if applied else None,
        replaced_tags=removed_tags if applied else (),
        detail=(
            "forced over local edits"
            if applied
            else (
                f"differs from the packaged starter in {', '.join(diverged)}"
                + (f" (installed {installed_version})" if installed_version else "")
                + ". It looks edited, so it was left alone. To discard your edits and take "
                "the packaged text: " + f"`lop agents sync --name {key} --replace --yes` "
                "(your current text is printed so you can save it)."
            )
        ),
    )


# -- startup pass ------------------------------------------------------------
#
# THE STARTUP HALF OF #2060, and where it lives: `run_startup_migrations` is
# the ONE seam `cli.main` calls for every subcommand, and its doctrine (never
# raise, no stamp file, refuse before constructing anything on a storeless
# machine) is exactly what an unattended write needs. One arm here is NOT
# enough by itself, because the seam knows about surfaces:

#: Whether the startup pass may auto-apply untouched starter updates.
#: Default ON: an operator upgrading ``lop`` reasonably expects the built-in
#: roles to learn what the release taught them (issue #2060), and the pass
#: only ever writes rows the LEDGER proves are unedited published revisions
#: that are strictly behind their packaged starter. The key
#: (``agents.auto_update.seeds``) turns the unattended write off while keeping
#: the drift REPORT: sync stays the only writer.
AUTO_UPDATE_SEEDS_DEFAULT = True

#: File name of the per-config-root notice queue, beside ``config.yml``. It is
#: deliberately IN the config dir (not the package): every config root gets
#: its own de-dup memory, and deleting it costs at most one repeated notice.
SEED_NOTICES_NAME = ".seed-notices.json"

#: Ceiling on queued TUI notice lines. A machine that never opens the TUI
#: (a pure-CLI or desktop-only install) would otherwise grow ``pending`` by a
#: few lines per upgrade forever; the oldest lines are the ones to drop.
MAX_PENDING_SEED_NOTICES = 50

#: The lock file name serialising unattended writes and notice-state updates.
#: Sized for a machine where several ``lop`` processes start at once (TUI,
#: serve, wake) — the same contention the wake store's lock already answers.
SEED_SYNC_LOCK_NAME = ".seed-sync.lock"


@dataclass(frozen=True)
class SeedStartupOutcome:
    """What one startup pass did, per row and per notice.

    A frozen record rather than log-only sides: the pass runs unattended, so
    the test suite is the only observer — and the fields are exactly what a
    test asserts without re-deriving state (an ``applied`` name, a held name,
    the lines it announced). ``skipped`` carries the rows whose write was
    withheld because the lock was busy; "silent" rows (no ledger proof,
    ahead, up-to-date) are deliberately absent from every field — silence is
    the documented outcome for them.
    """

    applied: tuple[str, ...] = ()
    held: tuple[str, ...] = ()
    available: tuple[str, ...] = ()
    announced: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()


def _auto_update_seeds_enabled(config_dir: Path) -> bool:
    """The ``agents.auto_update.seeds`` key, read once per pass. Never raises.

    Absent means the shipped default (ON) — turning the pass off is an
    explicit off. A non-bool value reads as unset rather than truthy (the
    ``hub_sync.settings._bool`` rule): a hand-edited ``"false"`` string must
    neither mean True nor crash the start path.
    """

    from local_operator.config import ConfigManager

    value = ConfigManager(config_dir).get_nested_value(
        ("agents", "auto_update", "seeds"), AUTO_UPDATE_SEEDS_DEFAULT
    )
    return value if isinstance(value, bool) else AUTO_UPDATE_SEEDS_DEFAULT


def _installed_via_updater() -> bool:
    """Whether THIS install kind may be self-updated unattended.

    Lazy import, and the call site late: ``local_operator.update`` drags
    ``ssl``/``urllib``/``http.client``, and the pass must not pay that on
    every CLI start. It is also the seam a harness patches to exercise the
    apply path from a dev venv: an EDITABLE install (a worktree ``.venv``) is
    report-only by design, because dev venvs share the operator's real config
    dir (finding F6 of #2060) and an edit in one must not rewrite a row the
    installed runtime is using.
    """

    from local_operator.update import InstallKind, install_kind

    return install_kind() in (InstallKind.UV_TOOL, InstallKind.PIPX, InstallKind.PIP)


def _seed_version_note(installed: str | None, packaged: str) -> str:
    """``" (1.0.0 -> 1.4.0)"`` for a notice line, or ``""`` when unknowable."""

    if installed and packaged:
        if installed == packaged:
            return f" ({packaged}, text revised)"
        return f" ({installed} -> {packaged})"
    return ""


def _seed_display_name(agent: "AgentData") -> str:
    """The row as the user sees it (``Aida``, not ``aida``)."""

    from local_operator.display_labels import display_form

    return display_form(str(agent.name or ""), str(getattr(agent, "label", "") or ""))


#: How a capability field reads in copy: ``(long, short)``. The long form is
#: for the held notice, the short one for the ``--check`` line. ONE table so
#: the notice and the render can never disagree about what ``tools`` means
#: (design round 1, D2).
_CAPABILITY_WORDS: dict[str, tuple[str, str]] = {
    "tools": ("the role's tool access", "tool access"),
    "effort": ("which model tier the role runs on", "the model tier the role runs on"),
    "delegate": ("what the role may delegate", "what the role may delegate"),
}


def _capability_note(diverged_fields: tuple[str, ...]) -> tuple[str, str]:
    """``(long, short)`` for the FIRST capability field in ``diverged_fields``.

    ``("", "")`` when no capability field diverged. First match wins because
    one note reads; the field list a refusal prints already names the rest.
    """

    for field, (long, short) in _CAPABILITY_WORDS.items():
        if field in diverged_fields:
            return long, short
    return "", ""


def _held_notice_line(display: str, name: str, diverged_fields: tuple[str, ...]) -> str:
    """The held-row notice: an update exists, but it changes a capability.

    The copy names ``--check`` FIRST for a reason: the old wording said \"to
    review\" while naming a command that applies the change outright, and the
    one surface that really previews (``--check``) never said what the held
    change was (design round 1, D2; UX round 1, U4). The word \"review\" may
    only name a command that really reviews.
    """

    long, _short = _capability_note(diverged_fields)
    return (
        f"{display}: a newer packaged starter is available but was not applied "
        f"automatically, because it changes {long or 'what the role is allowed to do'}. "
        f"See what changes with `lop agents sync --name {name} --check`; "
        f"apply it with `lop agents sync --name {name}`."
    )


def _reported_notice_line(display: str, installed: str | None, packaged: str) -> str:
    """The report-only notice (auto-update off, or an install that cannot self-update).

    Deliberately carries NO off-switch pointer (design round 2, D2-1, which
    moved it to the applied notices): this notice fires exactly when the
    pointer cannot be satisfied — the setting is already off, or this install
    cannot self-update at all — so it would name a control already in the
    state the reader wants. The APPLIED notice is the surprise the pointer
    exists for; it carries it instead.
    """

    return (
        f"{display}: an update to the packaged starter is available"
        f"{_seed_version_note(installed, packaged)}. Run `lop agents sync` to apply it."
    )


def _applied_notice_line(display: str, installed: str | None, packaged: str) -> str:
    """The applied notice. Only ever composed AFTER a successful write.

    Carries the off-switch pointer (design round 2, D2-1, moved here from the
    reported notice): an UNEXPECTED write is where a reader most needs to
    learn the channel exists, and the pointer is satisfiable while reading
    this line — the reported notice fires exactly when it is not.
    """

    return (
        f"{display}'s instructions updated to the packaged starter"
        f"{_seed_version_note(installed, packaged)}; your label, model and tags were kept. "
        "(stop auto-updates: /settings → Agents → Auto-update built-in roles)"
    )


def _edited_notice_line(display: str, name: str, installed: str, packaged: str) -> str:
    """The edited-and-moved notice: report-only, provable facts only.

    Fires ONLY when both versions are recorded and differ (the gate lives at
    the call site): a row with no install record or no version proof stays
    silent, because neither half of the sentence could be shown true. The
    remedy is the typed command, which REFUSES on an edited row - safe by
    construction (design round 1, D8's option; the operator's standing
    direction that edits are reported, not swallowed).
    """

    return (
        f"{display}: you have edited these instructions and the packaged starter "
        f"has moved ({installed} -> {packaged}). Your copy was left alone. "
        f"Run `lop agents sync --name {name}` to see your options."
    )


def _rollup_transition(installed: str | None, packaged: str) -> str:
    """One member's ``1.0.0 -> 1.4.0`` transition for a rolled-up notice."""

    if installed and packaged and installed != packaged:
        return f"{installed} -> {packaged}"
    if packaged:
        return f"{packaged}, text revised"
    return "text revised"


def _rollup_members(entries: list[tuple[str, str | None, str]]) -> str:
    """``Aida 1.0.0 -> 1.4.0, Coder 1.0.0 -> 1.2.0, and 7 more`` for one group."""

    members = [
        f"{display} {_rollup_transition(installed, packaged)}"
        for display, installed, packaged in entries
    ]
    return f"{', '.join(members[:2])}, and {len(members) - 2} more"


def _applied_rollup_line(entries: list[tuple[str, str | None, str]]) -> str:
    """The >2-applied roll-up: one line instead of one per role (D4/U2).

    Carries the off-switch pointer for the same reason the individual applied
    line does (design round 2, D2-1); the reported roll-up below does not.
    """

    return (
        f"Updated {len(entries)} built-in roles to the packaged text "
        f"({_rollup_members(entries)}); your labels, models and tags were kept. "
        "(stop auto-updates: /settings → Agents → Auto-update built-in roles)"
    )


def _reported_rollup_line(entries: list[tuple[str, str | None, str]]) -> str:
    """The >2-available roll-up (action needed, so it names the command).

    No off-switch pointer, for ``_reported_notice_line``'s reason (D2-1).
    """

    return (
        f"{len(entries)} built-in roles have updates available "
        f"({_rollup_members(entries)}). Run `lop agents sync` to apply them."
    )


def _edited_rollup_line(entries: list[tuple[str, str | None, str]]) -> str:
    """The >2-edited roll-up; individual lines stay for one or two (D4)."""

    return (
        f"{len(entries)} built-in roles you edited have newer packaged starters "
        f"({_rollup_members(entries)}). Your copies were left alone. "
        "Run `lop agents sync` to see your options."
    )


def _load_seed_notice_state(config_dir: Path) -> tuple[dict[str, str], list[str]]:
    """``(announced, pending)`` from ``.seed-notices.json``; corrupt reads empty.

    A corrupt or missing file costs at most ONE duplicate notice (the de-dup
    map is what is lost) — it can never suppress a repair or raise on the
    start path. That is why this is a legitimate exception to the seam's
    "no stamp file" doctrine: the doctrine exists because a stale/corrupt
    stamp could SKIP a migration; this file gates only display.
    """

    try:
        payload = json.loads((Path(config_dir) / SEED_NOTICES_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, []
    if not isinstance(payload, dict):
        return {}, []
    announced_raw = payload.get("announced")
    pending_raw = payload.get("pending")
    announced = (
        {str(key): str(value) for key, value in announced_raw.items()}
        if isinstance(announced_raw, dict)
        else {}
    )
    pending = [str(line) for line in pending_raw] if isinstance(pending_raw, list) else []
    return announced, pending


def _write_seed_notice_state(
    config_dir: Path, *, announced: dict[str, str], pending: list[str]
) -> None:
    """Replace ``.seed-notices.json`` atomically; a peer's concurrent write is
    last-writer-wins and costs at most a duplicate notice, never a partial
    file — readers only ever see whole documents (``os.replace``)."""

    from local_operator.agents import _write_text_atomically

    payload = {
        "schema_version": 1,
        "announced": announced,
        "pending": pending,
    }
    _write_text_atomically(
        Path(config_dir) / SEED_NOTICES_NAME,
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )


def peek_pending_seed_notices(config_dir: Path) -> list[str]:
    """The queued notice lines, WITHOUT clearing them.

    Half one of the delivery split (agent review round 1, R1-3): the TUI hook
    PEEKS, displays, and only then clears - so a crash between the two shows a
    duplicate next boot (safe direction) instead of losing the line (silent).
    Clearing first, which the old single-function drain did, loses a notice
    whenever the display half fails; its docstring then claimed the opposite.
    """

    _announced, pending = _load_seed_notice_state(config_dir)
    return pending


def clear_pending_seed_notices(config_dir: Path, displayed: list[str]) -> None:
    """Remove the DISPLAYED lines from ``pending``, keeping ``announced``.

    Only the lines actually shown are removed (by value): another process may
    have queued fresh ones between the peek and this call, and clearing the
    whole list would discard them unseen. A write failure leaves ``pending``
    in place - the next boot shows them again, the safe direction.
    """

    announced, pending = _load_seed_notice_state(config_dir)
    if not pending:
        return
    remaining = list(pending)
    for line in displayed:
        if line in remaining:
            remaining.remove(line)
    try:
        _write_seed_notice_state(config_dir, announced=announced, pending=remaining)
    except OSError:
        logger.debug("seed notices: could not clear pending; they will show again")


def startup_seed_update_pass(
    config_dir: Path,
    *,
    surface: str = "cli",
) -> SeedStartupOutcome:
    """Report starter drift on every launch; auto-apply the rows the ledger proves.

    THE #2060 HEADLINE, in the one place the whole startup doctrine allows an
    unattended writer. What it does, in order, and every refusal is silent by
    design except where a notice is named:

    * STORELESS MACHINES RETURN BEFORE ANYTHING IS CONSTRUCTED: no ``agents/``
      directory, no registry, no config read — ``AgentRegistry.__init__`` would
      ``mkdir`` the store the machine deliberately does not have
      (``test_cli_org_sharing`` pins that). The same rule
      ``backfill_seed_action_class`` documents.
    * A missing or corrupt LEDGER ends the pass silently: without positions
      there is no proof of direction, and a dev build must not nag.
    * A row split into: applied (ledger-positioned strictly behind, delta
      touching only prose/description/class — ``class`` is NOT a hold test, it
      is user data the narrow writer preserves (agent review round 1, R1-2);
      packaged text == ledger tail, this install kind may self-update, the
      setting is on — written through the narrow writer under a lock, after a
      re-check); held (ledger-positioned, but the delta touches
      ``tools``/``effort``/``delegate`` — a capability boundary is never
      widened unattended); available (eligible but report-only: the setting
      is off, this install kind is EDITABLE, or the launch is a DAEMON —
      a launch nobody reads must not write what nobody will be told about);
      edited (diverged from the
      packaged text with BOTH versions recorded and different — reported,
      never written); silent (up-to-date, ahead, no ledger proof, draft
      packaged text, rows with no version proof, unsupported rows).
    * NOTICES de-dup per SURFACE x seed x packaged revision in
      ``.seed-notices.json``, because a notice consumed by a log-only launch
      is a notice the person never sees (agent review round 1, R1-1; design
      D3). ``tui`` lines queue in ``pending`` for the boot hook; ``cli``
      lines print plainly to stderr (no ``date - INFO -`` prefix) and record
      only their own slot; ``daemon`` surfaces (``serve``, ``wake serve``,
      ``mobile serve``) are report-only: they write no row, record NOTHING
      and log at DEBUG, so the first human surface applies and announces.
      More than two rows in one group collapse to a
      single rolled line (design round 1, D4); held rows stay individual
      because they need action. ``--check``/``--dry-run`` never reach this
      file at all: the seam's carve-out keeps the whole pass off that
      command — and off ``config edit agents.auto_update.seeds``, whose
      whole point is to change the setting this pass reads (UX round 1,
      U6b).

    One writer at a time, bounded: the wake store's lock serialises peers (a
    machine starts the TUI, ``serve`` and a wake supervisor together), a busy
    lock skips this launch silently and the next retries, and the row is
    re-verified under the lock so a classification made moments earlier
    cannot write a row that moved since. Every failure is caught per row and
    per phase: the pass may skip, delay, or under-claim, and it may never
    raise into ``cli.main`` or claim a write it did not make.
    """

    config_dir = Path(config_dir)

    # (1) THE STORELESS GATE, before any construction. See the docstring.
    if not (config_dir / "agents").is_dir():
        return SeedStartupOutcome()

    revisions = load_seed_revisions()
    if not revisions:
        return SeedStartupOutcome()

    from local_operator.agents import AgentRegistry

    try:
        registry = AgentRegistry(config_dir)
        rows = _installed_seed_rows(registry)
    except Exception:  # noqa: BLE001 - never a reason a launch fails
        logger.debug("seed update pass: no readable registry", exc_info=True)
        return SeedStartupOutcome()

    if not rows:
        return SeedStartupOutcome()

    classified: list[tuple[str, SeedSyncVerdict]] = []
    for key in sorted(rows):
        try:
            classified.append(
                (
                    key,
                    _sync_one_seed(
                        registry, key, rows[key], force=False, apply=False, revisions=revisions
                    ),
                )
            )
        except Exception:  # noqa: BLE001 - one unreadable row never stops the pass
            logger.debug("seed update pass: skipping row %r", key, exc_info=True)

    apply_candidates: list[str] = []
    held: list[str] = []
    available: list[str] = []
    edited: list[str] = []
    for key, verdict in classified:
        if verdict.verdict == "outdated-diverged":
            # Reported when the person's copy is provably an edit AND the
            # package provably moved: both versions recorded, different. A
            # no-record row stays silent - neither half of the sentence could
            # be shown true (design round 1, D8's option).
            if (
                verdict.installed_version
                and verdict.packaged_version
                and verdict.installed_version != verdict.packaged_version
            ):
                edited.append(key)
            continue
        if verdict.verdict != "outdated-clean" or verdict.behind_by is None:
            # up-to-date, ahead, or no ledger position: silent. A stamp-cleaned
            # row lands here too - unattended writes require the ledger.
            continue
        if _packaged_tail_entry(key, revisions) is None:
            # The packaged text is not the ledger's tail: a draft prompt in a
            # worktree, or a rollback. Never auto-apply, never nag.
            continue
        if {"tools", "effort", "delegate"} & set(verdict.diverged_fields):
            # A capability boundary is never widened unattended. ``class`` is
            # deliberately NOT in this set: a switched class is user data the
            # narrow writer preserves, and holding on it left the reporter's
            # own row (``class: proactive``) without its update (agent review
            # round 1, R1-2 / ADR Q2).
            held.append(key)
        else:
            apply_candidates.append(key)

    if apply_candidates:
        if surface == "daemon":
            # A DAEMON LAUNCH NEVER WRITES, which is the closest correct reading
            # of "a daemon consumes nothing": a launch nobody reads cannot tell
            # anyone it rewrote a role, and the row is CURRENT by the time a
            # human surface runs - so the applied notice could never fire for
            # it (the apply IS the event; the human launch would classify the
            # row up-to-date and say nothing). Leaving the rows for the first
            # human surface keeps "every write is announced" true, and matches
            # the ADR's deferral of desktop auto-update together with its
            # notice. Report-only here: logged at DEBUG, nothing recorded.
            available.extend(apply_candidates)
            apply_candidates = []
        elif not _auto_update_seeds_enabled(config_dir):
            available.extend(apply_candidates)
            apply_candidates = []
        elif not _installed_via_updater():
            available.extend(apply_candidates)
            apply_candidates = []

    # A notice is a DISPLAY event: only the two HUMAN surfaces (``tui``, ``cli``)
    # consume one. Any other surface (a daemon, or a name this code does not
    # know) is report-only - it logs at DEBUG and neither writes a row nor
    # records a token, so the first human surface still applies and announces
    # (agent review round 1, R1-1 / design round 1, D3).
    human = surface in ("tui", "cli")

    def _notice_token(key: str) -> str | None:
        entry = _packaged_tail_entry(key, revisions)
        # SURFACE-scoped: a notice consumed by one surface's launch must not
        # silence another's (a log-only process spending the TUI's token was
        # exactly the R1-1 defect). ``None`` for a draft packaged text (not the
        # ledger's tail): never nag about a dev build's unpublished prompt.
        return f"{surface}:{key}:{entry.sha}" if entry is not None else None

    announced, pending = _load_seed_notice_state(config_dir)

    def _unannounced(keys: list[str]) -> list[str]:
        """The keys THIS surface has not yet been told about.

        A group's roll-up must list only these: re-listing rows an earlier
        launch already announced would repeat old news inside a new line the
        moment ONE seed gains a revision.
        """

        fresh: list[str] = []
        for key in keys:
            token = _notice_token(key)
            if token is None or rows.get(key) is None:
                continue
            if human and token in announced:
                continue
            fresh.append(key)
        return fresh

    planned: list[tuple[tuple[str, ...], str]] = []  # (tokens, line), emission order

    def _plan(keys: list[str], line: str) -> None:
        tokens = tuple(token for token in (_notice_token(key) for key in keys) if token)
        if tokens:
            planned.append((tokens, line))

    def _group_entries(keys: list[str]) -> list[tuple[str, str | None, str]]:
        """``(display, installed, packaged)`` per key, for roll-up member lines."""

        return [
            (
                _seed_display_name(rows[key]),
                marker_value(rows[key], SEED_VERSION_PREFIX),
                load_seed_version(key),
            )
            for key in keys
        ]

    verdicts_by_key = dict(classified)
    for key in _unannounced(held):
        # Held rows are ALWAYS individual: they need action (design round 1, D4).
        verdict = verdicts_by_key.get(key)
        _plan(
            [key],
            _held_notice_line(
                _seed_display_name(rows[key]),
                key,
                verdict.diverged_fields if verdict is not None else (),
            ),
        )

    fresh_available = _unannounced(available)
    if len(fresh_available) > 2:
        # One rolled line for a typical multi-role report; individual lines
        # stay for one or two (design round 1, D4).
        _plan(fresh_available, _reported_rollup_line(_group_entries(fresh_available)))
    else:
        for key in fresh_available:
            _plan(
                [key],
                _reported_notice_line(
                    _seed_display_name(rows[key]),
                    marker_value(rows[key], SEED_VERSION_PREFIX),
                    load_seed_version(key),
                ),
            )

    fresh_edited = _unannounced(edited)
    if len(fresh_edited) > 2:
        _plan(fresh_edited, _edited_rollup_line(_group_entries(fresh_edited)))
    else:
        for key in fresh_edited:
            verdict = verdicts_by_key.get(key)
            _plan(
                [key],
                _edited_notice_line(
                    _seed_display_name(rows[key]),
                    key,
                    verdict.installed_version if verdict and verdict.installed_version else "",
                    verdict.packaged_version if verdict else "",
                ),
            )

    if not apply_candidates and not planned:
        return SeedStartupOutcome(
            held=tuple(sorted(held)),
            available=tuple(sorted(available)),
        )

    if not human:
        # Report-only (a daemon): nothing to WRITE, so nothing to take the lock
        # for - log the lines and leave the state file, and every human
        # surface's tokens, untouched. ``apply_candidates`` is already empty
        # for such a surface (see above).
        for _tokens, line in planned:
            logger.debug("seed update: %s", line)
        return SeedStartupOutcome(
            held=tuple(sorted(held)),
            available=tuple(sorted(available)),
        )

    from local_operator.wakes.lock import (
        WakeLockBusy,
        WakeLockUnavailable,
        WakeWriteLock,
    )

    # 2.0 s, not 5: a contended launch must not stall a TUI/CLI start for
    # seconds; the lock is held for milliseconds in practice (UX round 1, O2).
    lock = WakeWriteLock(config_dir, name=SEED_SYNC_LOCK_NAME, timeout_s=2.0)
    try:
        lock.acquire()
    except (WakeLockBusy, WakeLockUnavailable):
        # Another launch is mid-write: skip this launch SILENTLY (nothing
        # logged, nothing queued, nothing recorded); the next launch retries.
        logger.debug("seed update pass: another writer holds the lock; skipping")
        return SeedStartupOutcome(
            skipped=tuple(sorted({*apply_candidates, *held, *available, *edited}))
        )

    try:
        applied: list[str] = []
        # (key, display, installed-before, packaged, token): the PRE-write
        # values, captured because ``update_agent`` mutates the same ``agent``
        # object - reading the markers afterwards printed "(1.4.0, text
        # revised)" for a real 1.0.0 -> 1.4.0 jump (agent review round 1,
        # R1-3 / QA Q1 / design D1 / UX U1).
        apply_entries: list[tuple[str, str, str | None, str, str | None]] = []
        if apply_candidates:
            try:
                registry.refresh_now()
                fresh_rows = _installed_seed_rows(registry)
            except Exception:  # noqa: BLE001
                fresh_rows = rows
            for key in apply_candidates:
                agent = fresh_rows.get(key)
                if agent is None:
                    continue
                try:
                    # Re-verify under the lock: the row may have moved since
                    # classification (another process, a person's edit).
                    recheck = _sync_one_seed(
                        registry, key, agent, force=False, apply=False, revisions=revisions
                    )
                    if recheck.verdict != "outdated-clean" or recheck.behind_by is None:
                        continue
                    if _packaged_tail_entry(key, revisions) is None:
                        continue
                    seed = load_seed(key)
                    if seed is None:  # pragma: no cover - tier 1 already loaded it
                        continue
                    display_before = _seed_display_name(agent)
                    version_before = marker_value(agent, SEED_VERSION_PREFIX)
                    _apply_seed_update(registry, agent, seed)
                    applied.append(key)
                    apply_entries.append(
                        (
                            key,
                            display_before,
                            version_before,
                            load_seed_version(key),
                            _notice_token(key),
                        )
                    )
                except Exception:  # noqa: BLE001 - never partial claims, never stops start
                    logger.warning("seed update pass: could not update %r", key, exc_info=True)

        # RE-READ THE NOTICE STATE UNDER THE LOCK. The read above was only for
        # planning; writing a stale snapshot back would let this launch erase a
        # peer's freshly queued ``pending`` lines (a TUI starting beside a CLI
        # command is the ordinary case).
        announced, pending = _load_seed_notice_state(config_dir)

        # APPLIED lines are never de-duplicated against ``announced``. An apply
        # is a WRITE and must always be told; it cannot repeat on its own (the
        # row is current afterwards), but it CAN share a token with an earlier
        # "update available" line for the same revision - the person flips the
        # setting on after being told, and the apply must not then be silent.
        applied_lines: list[tuple[tuple[str, ...], str]] = []
        if len(apply_entries) > 2:
            # One rolled line replaces the applied group when more than two
            # rows moved (design round 1, D4; UX round 1, U2).
            members = [
                (display, installed, packaged)
                for _k, display, installed, packaged, _t in apply_entries
            ]
            applied_lines.append(
                (
                    tuple(token for *_, token in apply_entries if token),
                    _applied_rollup_line(members),
                )
            )
        else:
            for _key, display, installed, packaged, token in apply_entries:
                applied_lines.append(
                    (
                        (token,) if token is not None else (),
                        _applied_notice_line(display, installed, packaged),
                    )
                )

        emitted: list[str] = []
        applied_emitted: list[str] = []
        for tokens, line in applied_lines:
            for token in tokens:
                announced[token] = line
            emitted.append(line)
            applied_emitted.append(line)
        for tokens, line in planned:
            # Planned before the lock; a peer may have announced some since.
            fresh_tokens = tuple(token for token in tokens if token not in announced)
            if not fresh_tokens:
                continue
            for token in fresh_tokens:
                announced[token] = line
            emitted.append(line)

        if emitted:
            queue = list(pending)
            if surface == "tui":
                queue.extend(emitted)
            else:
                # A ``cli`` launch PRINTS its lines, but an APPLIED notice is a
                # one-time event the TUI could never re-derive (the row is
                # current by the time it opens), so it is ALSO queued for the
                # next TUI boot. Report-style lines need no such echo: the TUI
                # surface announces those itself under its own token.
                queue.extend(applied_emitted)
            try:
                _write_seed_notice_state(
                    config_dir, announced=announced, pending=queue[-MAX_PENDING_SEED_NOTICES:]
                )
            except Exception:  # noqa: BLE001 - display-only state
                logger.debug("seed update pass: could not record notices", exc_info=True)
            if surface == "cli":
                # Plainly to stderr, without the ``date - INFO -`` prefix: the
                # CLI's own message idiom; the log prefix read as chatter
                # (design round 1, D3's cosmetic note).
                for line in emitted:
                    print(line, file=sys.stderr)
        return SeedStartupOutcome(
            applied=tuple(applied),
            held=tuple(sorted(held)),
            available=tuple(sorted(available)),
            announced=tuple(emitted),
        )
    finally:
        lock.release()


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
