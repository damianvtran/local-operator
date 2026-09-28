"""Her display name: one reader, one rule set, and the sync both ways.

WHY THIS MODULE EXISTS. The chief of staff ships as "Aida", but she is the
operator's — they rename her. The 2026-09-28 report was that renaming her
PINNED CONVERSATION changed the conversation and nothing else: every other
surface (the ``/aida`` receipts, the desktop payload, the greeting) still
said "Aida", because each one carried its own copy of the string. The fix is
one config key, ``aida.name``, that every surface READS, plus the two write
paths that keep it in step with her conversation's title:

- a rename landing on her session (``Session.set_conversation_name``, the one
  writer every rename gesture funnels through — the TUI's ``/title``, the
  runtime's ``/rename``, the desktop, the phone) calls
  :func:`sync_config_name_from_title`, so ``aida.name`` follows the title;
- a config change (``/aida rename`` here, ``/settings`` there, the desktop
  elsewhere) reaches her live session through the config watcher, whose
  ``_aida_reconcile_now`` calls :func:`display_name` and re-titles the session
  in place; for a session nobody has open, :func:`reconcile_session_title`
  writes the new title to disk from :func:`local_operator.aida.bootstrap.
  ensure_session`.

CONFIG IS CANONICAL (design note, 2026-09-28). When the two disagree — a
half-finished rename, a hand-edited file — the config key wins and the session
title is rewritten to match, because the config key is the thing every surface
(including the desktop payload's ``name`` field) reads directly. The session
rename path only ever writes the two together, so the disagreement case is a
crash window, not a steady state.

ONE RULE SET, IN THE SETTINGS REGISTRY. What counts as a valid name (trimmed,
non-empty, ≤ :data:`MAX_NAME_CHARS`, no control characters) is enforced by the
``aida.name`` row's ``validate_value`` in ``settings_io`` — the same funnel
``/settings``, ``lop config edit`` and the PATCH route write through — and
:func:`validate_name` here delegates to it rather than restating the bounds.
The registry keeps its bounds as LITERALS (it deliberately stays off this
package's import path), and ``tests/unit/aida/test_aida_naming.py`` pins the
literals against this module's constants, the way ``test_settings_io`` pins
the defaults.

READS NEVER FAIL. Every reader here answers the default on an absent key, an
unreadable config, or a stored value that cannot be a name — the same
"consumers clamp silently" posture ``aida.cadence.at`` documents — because a
display name must not be able to break a boot, a receipt or a payload. Writes
raise (``ValueError`` with the refusal), so the surface that asked for the
rename can show the user what was wrong.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: The packaged name: what she is called when ``aida.name`` is unset, and the
#: value ``bootstrap.SESSION_TITLE`` is built from. A module-level constant
#: next to the code that reads the key, per the adding-a-configuration-key
#: rules; ``settings_io``'s registry row carries the same value as a literal
#: and ``test_settings_io``'s ``_consumer_defaults`` pins the two together.
DEFAULT_NAME = "Aida"

#: The longest a display name may be. Equal to ``session.naming.
#: MAX_TITLE_CHARS`` (80) ON PURPOSE: a session rename syncs its title into
#: this key, so the two caps have to agree or a title the session layer
#: accepted could be refused by the config layer mid-sync.
MAX_NAME_CHARS = 80


def normalize_name(value: str) -> str:
    """Collapse whitespace and trim — the canonical form every reader shows.

    Shared with the session title's own cleaning (``ConversationName.set``
    runs the same ``" ".join(text.split())``), so a name that travels
    session→config→session cannot drift through re-normalization.
    """
    return " ".join(value.split())


def display_name(config_dir: Path | str | None = None) -> str:
    """The configured display name; :data:`DEFAULT_NAME` when it cannot be used.

    Never raises — this is the reader every surface calls (TUI receipts, the
    desktop payload, the greeting), and a display name that could break a
    rendering path would be a worse bug than a wrong one. An absent key means
    the default; so does an unreadable config, a non-string value, or a stored
    string that fails :func:`validate_name` (a hand-edited ``config.yml``
    reaches no writer's validation).
    """
    try:
        from local_operator.config import ConfigManager
        from local_operator.paths import config_dir as resolve_config_dir

        root = Path(config_dir) if config_dir is not None else resolve_config_dir()
        raw = ConfigManager(config_dir=root).get_nested_value(("aida", "name"), DEFAULT_NAME)
    except Exception:  # noqa: BLE001 — a display name must not break a read path
        logger.debug("aida: could not read aida.name; using the default", exc_info=True)
        return DEFAULT_NAME
    if not isinstance(raw, str):
        return DEFAULT_NAME
    try:
        return validate_name(raw)
    except (ValueError, TypeError):
        logger.warning("aida: aida.name=%r is not a usable name; using %r", raw, DEFAULT_NAME)
        return DEFAULT_NAME


def validate_name(value: Any) -> str:
    """Return the canonical name, or raise ``ValueError`` with the refusal.

    The bounds live in the ``aida.name`` registry row (see the module
    docstring); this is the thin adapter the rename paths call so the TUI's
    receipt and the settings page's error show the SAME sentence for the same
    value.
    """
    from local_operator import settings_io

    setting = settings_io.BY_KEY.get("aida.name")
    if setting is None:  # pragma: no cover - registry guarantee, pinned by tests
        raise ValueError("aida.name is not registered in settings_io")
    problem = settings_io.validate(setting, value)
    if problem is not None:
        raise ValueError(problem)
    return normalize_name(value)


def set_name(config_dir: Path | str, value: Any) -> str:
    """Validate and store ``aida.name``; returns the canonical name.

    Raises ``ValueError`` (the refusal) when ``value`` cannot be a name. The
    write goes through ``settings_io.write_setting`` — the path the config
    watcher diffs, so a live session in ANY process hears the change on its
    next tick or the writer's own notify — and is skipped when the canonical
    value is already in force, so a no-op rename cannot ring the watcher back.
    """
    from local_operator import settings_io
    from local_operator.config import ConfigManager

    root = Path(config_dir)
    normalized = validate_name(value)
    setting = settings_io.BY_KEY["aida.name"]
    if normalized == display_name(root):
        return normalized
    settings_io.write_setting(ConfigManager(config_dir=root), setting, normalized)
    return normalized


def sync_config_name_from_title(config_dir: Path | str, title: str) -> str | None:
    """Adopt a user-set session title as ``aida.name``; best-effort, never raises.

    Called by ``Session.set_conversation_name`` when the renamed conversation
    is HERS. ``None`` when the title cannot be a display name (empty,
    over-long, control characters) — the session rename itself still stands;
    only the config half is skipped, because the config schema is what refuses
    such values everywhere and a decoration must never cost the rename.
    """
    try:
        return set_name(config_dir, title)
    except (ValueError, TypeError) as error:
        logger.warning("aida: not adopting %r as aida.name: %s", title, error)
        return None
    except Exception:  # noqa: BLE001 — a sync is decoration; the rename stands
        logger.warning("aida: could not sync aida.name from the session title", exc_info=True)
        return None


async def reconcile_session_title(
    config_dir: Path | str, session_id: str, *, name: str | None = None
) -> str | None:
    """Point her stored conversation title at the configured name.

    The NOT-LIVE half of the config-canonical rule: a rename made while her
    session has no open runtime (a ``/settings`` edit, ``/aida rename`` from a
    terminal that is not sitting on her conversation, the desktop while she is
    closed) must still reach the picker and the sidebar, which read the title
    from DISK. Writes both channels the live rename writes, in the same order
    ``Session._persist_conversation_name`` uses — the transcript's
    ``conversation_name`` entry first (what a resume restores), then the title
    sidecar (what the picker reads O(1)) — so the two records cannot disagree
    about a conversation.

    Best-effort by contract: ``None`` when the session directory is missing,
    or when any part of the write fails; the name in force otherwise. Never
    raises — both callers (``bootstrap.ensure_session``, the TUI's rename
    worker) run where a decoration failure must be invisible.
    """
    root = Path(config_dir)
    if name is None:
        wanted = display_name(root)
    else:
        try:
            wanted = validate_name(name)
        except (ValueError, TypeError):
            return None
    directory = root / "sessions" / session_id
    try:
        if not directory.is_dir():
            return None
        from local_operator.resume import (
            read_title_names,
            stored_session_title,
            write_session_title,
        )
        from local_operator.session.naming import CONVERSATION_NAME_CUSTOM_TYPE
        from local_operator.session.transcript import Transcript

        if stored_session_title(directory) == wanted:
            return wanted
        await Transcript(directory).append_custom(
            CONVERSATION_NAME_CUSTOM_TYPE, {"text": wanted, "user_set": True}
        )
        write_session_title(
            directory, wanted, user_set=True, past_names=read_title_names(directory)
        )
        return wanted
    except Exception:  # noqa: BLE001 — a title is decoration; never fail the caller
        logger.warning("aida: could not reconcile her stored title", exc_info=True)
        return None


def is_her_session(session: object) -> bool:
    """Whether ``session`` is HERS, whoever is asking.

    The rename receipts and the config sync must agree about when the "she is
    now called X everywhere" clause is true (UX round 1, U3); ``_aida_duty``
    stays the single source — the same gate the sync itself reads — and this
    accessor only spares each caller its own private-attribute walk. Read
    defensively: a reduced facade need not carry the flag, and "not hers" is
    the answer that changes no wording.
    """
    return bool(getattr(session, "_aida_duty", False))
