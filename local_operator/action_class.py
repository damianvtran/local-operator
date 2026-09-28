"""The ACTION CLASS of an agent — ``reactive`` (default) or ``proactive``.

WHAT A CLASS IS. An agent's class decides whether it may run the platform's
proactive mechanisms: patience waits (a hidden internal timer attached to a
sent message, cycled under hard bounds — see ``local_operator/wakes/patience``)
and engine-armed proactive deliveries (Aida's cadence). ``reactive`` is today's
behaviour exactly: the class is a GENERAL mechanism, not an Aida flag, so any
agent may be switched to ``proactive`` and the user-facing toggle is a later
surface; Aida is simply the agent that ships in the proactive class.

WHERE IT LIVES IN DATA. ``AgentProfile.action_class``, parsed from a seed's
``class:`` frontmatter and encoded on installed registry rows as a
``class:proactive`` tag — the exact precedent ``delegate:yes`` / ``tools:…`` /
``effort:…`` already set (``agent_profiles.seed_tags`` / ``profile_from_agent``;
the tag rides in ``AgentData.tags`` so no schema migration is needed). Python
cannot name an attribute ``class``, hence ``action_class`` in code; the
user-facing word stays "class". The encoding is deliberately SPARING —
``reactive`` is the ABSENT tag, so a row with no class tag reads reactive and
old rows need no migration.

THE EFFECTIVE CLASS OF A SESSION is its attached profile's class, read through
the existing attachment-restore path: the ``attachment.json`` sidecar names the
profile, ``resolve_profile`` resolves it (registry first, packaged seed
second), and this module returns the resolved class. No attachment ⇒ reactive.
Crucially this is a DELIVERY-TIME read (:func:`session_action_class`): every
proactive path asks at the moment it acts rather than caching at session build,
so switching an agent's class takes effect on a RUNNING session at its next
decision point (R36) — a switch away from ``proactive`` stops the cadence and
cancels nothing else; ordinary chat is unaffected.

FAIL CLOSED, deliberately, and it is the opposite of the Aida hold's fail-open.
An unreadable class in front of a patience fire or an engine-arm means "do not
be proactive": a stop the user asked for that keeps messaging is the nagging
bug the class exists to end, while a bounded wait that ends a little early
costs one re-arm (R34 already attaches a fresh wait on the next reply). The
failure is logged, never raised — the callers are delivery paths that must
degrade, not crash.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

#: The class every agent has unless it says otherwise. Kept as the literal
#: spelling in one place so the tag encoder and every reader agree.
REACTIVE = "reactive"
#: The class that unlocks patience waits and engine-armed proactive deliveries.
PROACTIVE = "proactive"

VALID_CLASSES: tuple[str, ...] = (REACTIVE, PROACTIVE)

#: The tag key that carries the class on an installed registry row. The
#: ``key:value`` shape is shared with ``tools:``/``effort:``/``delegate:``.
TAG_KEY = "class"


def normalize(value: object, *, default: str = REACTIVE) -> str:
    """Coerce a raw class spelling to a valid class, or ``default``.

    Case/whitespace tolerant because both sides of the wire are human-edited
    in places (a seed's frontmatter, a registry tag); anything unrecognized,
    including ``None`` and ``""``, answers the default rather than raising —
    the callers are delivery gates that must keep working.
    """
    text = str(value or "").strip().lower()
    return text if text in VALID_CLASSES else default


def is_class_tag(tag: object) -> bool:
    """Whether one tag is a ``class:`` tag (case-insensitive key)."""
    key, sep, _ = str(tag).partition(":")
    return bool(sep) and key.strip().lower() == TAG_KEY


def class_from_tags(tags: Iterable[object] | None) -> str:
    """The class encoded in an installed row's tags; absent ⇒ reactive.

    First tag wins, mirroring ``profile_from_agent``'s read loop; the
    ``seed_tags`` writer emits at most one class tag, so a duplicate can only
    come from a hand-edited row and "first" is as good an answer as "last".
    """
    for tag in tags or ():
        if is_class_tag(tag):
            _, _, value = str(tag).partition(":")
            return normalize(value)
    return REACTIVE


def with_class_tag(tags: Iterable[object] | None, action_class: object) -> tuple[str, ...]:
    """``tags`` with exactly one ``class:`` tag carrying ``action_class``.

    ``reactive`` REMOVES the tag: absent means reactive, so the encoding stays
    sparing and a row that has never been proactive carries no class at all.
    Every other tag — including provenance markers and the role tag — rides
    through untouched, because this helper's whole job is the class half; a
    caller that wants to rebuild a profile's fields uses ``seed_tags``.
    """
    kept = [str(tag) for tag in (tags or ()) if not is_class_tag(tag)]
    resolved = normalize(action_class)
    if resolved == PROACTIVE:
        kept.append(f"{TAG_KEY}:{PROACTIVE}")
    return tuple(kept)


def session_action_class(session_dir: Path | str, *, registry: Any = None) -> str:
    """The EFFECTIVE class of a session, read at the moment it is asked.

    The attachment sidecar names the attached profile; ``resolve_profile``
    resolves the name (registry first, packaged seed second — the same order
    ``attach_agent_profile`` uses, so the class a session reports can never
    disagree with the profile it actually attached). Anything unresolved —
    no sidecar, unknown name, unreadable file, a profile with no class — reads
    ``reactive``: see the module docstring for why this direction is the safe
    one, and note it is a different posture from the Aida hold's fail-open.

    Never raises. The caller is a delivery path (a patience fire deciding
    whether to speak, an arm deciding whether it may exist); degrading to
    reactive is the answer that can never nag.
    """
    try:
        from local_operator.agent_profiles import resolve_profile
        from local_operator.resume import read_session_attachment

        stored = read_session_attachment(Path(session_dir))
        name = str(getattr(stored, "agent", "") or "").strip() if stored is not None else ""
        if not name:
            return REACTIVE
        profile = resolve_profile(name, registry=registry)
        if profile is None:
            return REACTIVE
        return normalize(profile.action_class)
    except Exception:  # noqa: BLE001 — documented fail-closed, see docstring
        logger.warning("could not resolve a session's action class; reading reactive", exc_info=True)
        return REACTIVE


def set_registered_action_class(registry: Any, name: str, action_class: object) -> str:
    """Flip one agent's class IN PLACE. Returns the row's name.

    The switch's storage half, shared by the ``agent`` tool, the TUI
    ``/agent class`` subcommand and the desktop profile route, so the tag
    rewrite cannot be spelled two ways. Only the class tag is touched:
    ``seed_tags`` would rebuild the whole tag list from a profile and is the
    wrong tool for a flip (it would need the full profile round-tripped
    through a write just to change one tag).

    A PACKAGED, NOT-YET-INSTALLED starter is switched too: it is installed
    first through the ordinary ``install_seed`` path (which writes the seed's
    tags, class included), then the flip applies. Without this, "switch Aida to
    reactive" — a seeded role whose session attaches the seed — would have
    nowhere to put the tag and would refuse the very switch R36 promises.

    Raises ``ValueError`` with a model-readable sentence when the name is not
    an installable/installed agent or the class is not valid — a typo must be
    reported, not half-applied. The registry stays the only storage authority;
    this helper only computes the new tag list.
    """
    from local_operator.agent_profiles import is_role, is_specialist
    from local_operator.agents import AgentEditFields

    resolved = str(action_class or "").strip().lower()
    if resolved not in VALID_CLASSES:
        raise ValueError(f"class must be one of {', '.join(VALID_CLASSES)}; got {action_class!r}.")
    key = (name or "").strip()
    if not key:
        raise ValueError("name an agent to switch; /agent lists them.")

    def _row() -> Any:
        try:
            return registry.get_agent_by_name(key)
        except Exception:  # noqa: BLE001 — a registry read failure is "not found"
            return None

    agent = _row()
    if agent is None or not (is_role(agent) or is_specialist(agent)):
        # Either nothing by that name, or a row the resolvers treat as an
        # ordinary agent (not attachable, not switchable). A packaged seed can
        # be materialized; anything else is a refusal.
        try:
            from local_operator.agent_profiles import install_seed

            installed = install_seed(key, registry=registry)
        except Exception as error:  # noqa: BLE001 — NameTakenError et al are refusals
            raise ValueError(f"cannot switch {name!r}: {error}") from None
        if installed is None:
            raise ValueError(f"no agent named {name!r} to switch class on.")
        agent = _row()
    if agent is None or not (is_role(agent) or is_specialist(agent)):
        raise ValueError(f"no installed role named {name!r} to switch class on.")

    tags = list(with_class_tag(agent.tags, resolved))
    # Every field spelled out: ``AgentEditFields`` is validated in strict mode
    # (the same convention ``write_profile`` documents).
    registry.update_agent(
        agent.id,
        AgentEditFields(
            name=None,
            description=None,
            tags=tags,
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
        ),
    )
    return str(agent.name)
