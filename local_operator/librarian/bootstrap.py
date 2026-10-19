"""The Librarian's bootstrap: ensure her single session exists, pinned, armed.

:func:`ensure_session` is the ONE place her session directory is created,
called from the TUI boot, the server lifespan and the desktop route's
``open``/``run`` ops — every entry point routes through it, so two concurrent
first invocations cannot mint two conversations (``ensure.lock`` serialises
them and the state file is re-read under it; the loser returns the id the
winner wrote).

HOW THE SESSION IS CREATED, and why not by constructing one. The Aida
bootstrap's reasoning applies unchanged: constructing a Session needs a
resolvable provider+model and does real work (MCP wiring, skills,
classification) an empty organizing conversation has no use for — and the
Librarian must be creatable on an install with NO provider configured, where
the factory raises. So this writes the durable facts a session directory is
made of, in this order:

1. mint a session id (``uuid4().hex[:12]`` — the shape every other mint uses);
2. ``mkdir(sessions/<id>, mode=0o700)`` — ``exist_ok=False`` under the lock: a
   collision means the id was not fresh and must retry loudly;
3. ``ensure_session_created_at`` — the same created-at sidecar the factory
   writes, so the picker's two clocks work on her row;
4. ``write_session_attachment(agent="librarian")`` — the sidecar the load path
   restores as her attached role. The packaged seed resolves through
   ``resolve_profile`` even uninstalled, so nothing user-authored is required;
5. the conversation title, both channels: the ``conversation_name`` custom
   entry (the transcript reader's fallback) and the title sidecar (the O(1)
   picker read), ``user_set=True`` so the auto-namer may not replace it after
   her first turn — the configured ``librarian.name`` is what the row says;
6. a birth custom entry, which is ALSO the reason the directory becomes REAL:
   ``session_activity`` returns ``None`` for a directory holding neither a
   transcript nor a mail spool, and such a directory is neither listed nor
   resumable — precisely the state the route's open→resume handoff and the
   picker row must not be in;
7. write ``state.json`` — AFTER the directory exists, so a crash in between
   leaves an orphan directory (invisible: no transcript) rather than a state
   file pointing at nothing;
8. pin her (``sidebar_pins.set_pin``, desired-state, idempotent) so the
   sidebar's pinned-first ordering holds from the first frame;
9. arm the cadence through :func:`local_operator.librarian.proactive.ensure_armed`
   — the transcript-first external writer, which also installs the wake
   supervisor. Steps 1-8 build the session; THIS step runs on EVERY call, for
   an existing id as well as a new one, because "she has a cadence" is not a
   fact a creation can establish once and forget: any later loss of the row
   would otherwise be permanent and silent;
10. reap the rows the PREVIOUS incarnation left on the wake index — a
    re-creation mints a new id, and the old id's rows can never fire.

NO ``origin.json``, DELIBERATELY. Aida's visibility rule is that a bespoke
origin is hidden by ``resume.is_user_session_origin`` and therefore invisible
on the phone; the Librarian is a session the operator can open, so she carries
the ordinary user origin (none at all) plus the attachment. That is what makes
her show up on every surface without a new tab (§5).

THE LOCK IS HELD ACROSS THE CREATE on the documented wake-lock pattern: the
acquire/release hop to a worker thread so the event loop stays free, and the
work between them is a handful of file writes (no model, no provider). A
contended lock is answered with ``None`` and a quiet log line — every caller
is best-effort by design, and the next boot retries.

DISABLED. The gate runs first — before the lock, before any path is joined —
and answers ``None`` with zero writes.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
import uuid
from pathlib import Path

from local_operator.librarian import proactive, state
from local_operator.wakes.lock import WakeLockBusy, WakeLockUnavailable

logger = logging.getLogger(__name__)

#: The master switch's default. Defined by the engine (its reader) and
#: re-exported here so the boot gate and the engine share one value; the
#: settings anti-drift test reads it from here for the ``librarian.enabled``
#: row.
DEFAULT_ENABLED = proactive.DEFAULT_ENABLED

#: The packaged role's name: the seed file (``agent_seeds/librarian.md``), the
#: attachment sidecar's ``agent`` field, and the name ``/agent librarian``
#: resolves.
ROLE_NAME = "librarian"

#: The packaged title's default; creation writes ``librarian.name``.
SESSION_TITLE = "Librarian"

#: The birth custom entry's type. Never enters the model's context; it is the
#: activity clock's reason to rank her row and a permanent marker of how the
#: conversation was born.
BIRTH_CUSTOM_TYPE = "librarian_session"

#: A display name is a label, not a document. Kept equal to the settings layer's
#: ``_AIDA_NAME_MAX_CHARS`` / the registry's ``_LIBRARIAN_NAME_MAX_CHARS``; the
#: anti-drift test pins the three together.
MAX_NAME_CHARS = 80


def display_name(config_dir: Path | str | None = None) -> str:
    """The configured ``librarian.name``, never empty and never raising.

    Read through a fresh ``ConfigManager`` with the same never-raise posture
    every boot-path config read uses. A blank/whitespace value, a non-string or
    a value past :data:`MAX_NAME_CHARS` falls back to the packaged default
    rather than crashing a surface that only wanted a label.
    """
    try:
        from local_operator.config import ConfigManager
        from local_operator.paths import config_dir as resolve_config_dir

        root = Path(config_dir) if config_dir is not None else resolve_config_dir()
        raw = ConfigManager(config_dir=root).get_nested_value(("librarian", "name"), SESSION_TITLE)
    except Exception:  # noqa: BLE001 — a label must never fail its surface
        logger.warning("librarian: could not read librarian.name; using the default")
        return SESSION_TITLE
    text = raw.strip() if isinstance(raw, str) else ""
    if not text:
        return SESSION_TITLE
    return text[:MAX_NAME_CHARS]


def config_enabled(config_dir: Path | str | None = None) -> bool:
    """Whether the Librarian is enabled: the config key AND the env switch.

    ``librarian.enabled`` (default true) is read through a fresh
    ``ConfigManager`` with the same never-raise posture every boot-path config
    read uses; the env switch is :func:`local_operator.librarian.state.env_disabled`.
    Either one disables. Import-light: ``local_operator.config`` is imported
    lazily.
    """
    if state.env_disabled():
        return False
    try:
        from local_operator.config import ConfigManager
        from local_operator.paths import config_dir as resolve_config_dir

        root = Path(config_dir) if config_dir is not None else resolve_config_dir()
        raw = ConfigManager(config_dir=root).get_nested_value(
            ("librarian", "enabled"), DEFAULT_ENABLED
        )
    except Exception:  # noqa: BLE001 — an unreadable config must not enable-or-crash
        logger.warning("librarian: could not read librarian.enabled; treating as enabled")
        return True
    return bool(raw)


def _sessions_root(config_dir: Path) -> Path:
    return Path(config_dir) / "sessions"


async def _create_session_dir(config_dir: Path, session_id: str) -> None:
    """Write the durable facts of a brand-new session. Raises on failure."""
    from local_operator.resume import write_session_attachment, write_session_title
    from local_operator.session.creation import ensure_session_created_at
    from local_operator.session.transcript import Transcript

    directory = _sessions_root(config_dir) / session_id
    directory.mkdir(parents=True, mode=0o700)
    # THE CONFIGURED NAME, not the packaged default: a session first created
    # AFTER the operator renamed her must be born wearing that name — the
    # sidebar's pinned row and the picker read this title, and they cannot ask
    # a session that does not exist yet.
    title = display_name(config_dir)
    ensure_session_created_at(directory, time.time())
    write_session_attachment(directory, team="", agent=ROLE_NAME, goal="")
    write_session_title(directory, title, user_set=True, past_names=[])
    transcript = Transcript(directory)
    # The name entry rides BEFORE the birth entry so the transcript reader's
    # first window already holds the title if the sidecar was lost.
    from local_operator.session.naming import CONVERSATION_NAME_CUSTOM_TYPE

    await transcript.append_custom(CONVERSATION_NAME_CUSTOM_TYPE, {"text": title, "user_set": True})
    await transcript.append_custom(
        BIRTH_CUSTOM_TYPE,
        {"created_at_ms": int(time.time() * 1000), "schema_version": state.STATE_SCHEMA},
    )


def _discard_failed_create(config_dir: Path, session_id: str) -> None:
    """Best-effort removal of a directory this call created but could not finish.

    Guarded hard: only when the directory exists and holds NO transcript (the
    window this covers is between ``mkdir`` and the first append — nothing
    durable can be inside it), so it can never delete a real conversation.
    """
    from local_operator.resume import TRANSCRIPT_NAME

    directory = _sessions_root(config_dir) / session_id
    try:
        if not directory.is_dir() or (directory / TRANSCRIPT_NAME).exists():
            return
        shutil.rmtree(directory)
    except OSError:
        logger.debug("librarian: could not clean up a failed session create", exc_info=True)


def _reconcile_session_title(config_dir: Path, session_id: str) -> None:
    """Bring an existing session's stored title to ``librarian.name``.

    Cheap when nothing moved: one small sidecar read and a compare. Best-effort
    — a title is decoration, and a rename made while she has no runtime must
    still reach the picker and the sidebar, which read the sidecar from disk.
    """
    try:
        from local_operator.resume import stored_session_title, write_session_title

        directory = _sessions_root(config_dir) / session_id
        wanted = display_name(config_dir)
        current = stored_session_title(directory)
        if current and current != wanted:
            write_session_title(directory, wanted, user_set=True, past_names=[current])
    except Exception:  # noqa: BLE001 — a label never fails an ensure
        logger.debug("librarian: could not reconcile the session title", exc_info=True)


async def ensure_session(
    config_dir: Path | str | None = None, *, now_ms: int | None = None
) -> str | None:
    """Her session id, creating the session on first need. ``None`` when disabled.

    Idempotent on every call after the first, and deliberately not free: one
    state read, one directory stat, one title reconcile, one wake-index read to
    answer "is her cadence armed?" and one listing of the index to reap any
    orphaned engine rows. What it never does is construct a session or resolve
    a provider — the two costs that would make a boot hook expensive. Safe to
    call from boot paths (best-effort: any failure logs and answers ``None``)
    and from every ``open``/``run`` op.
    """
    from local_operator.paths import config_dir as resolve_config_dir

    root = Path(config_dir) if config_dir is not None else resolve_config_dir()
    if not config_enabled(root):
        return None

    lock = state.wake_lock(root)
    try:
        await asyncio.to_thread(lock.acquire)
    except (WakeLockBusy, WakeLockUnavailable) as exc:
        # A refusal, not a defect: the shared shape keeps a stack out of the
        # operator's log. The next ensure tries again.
        state.note_lock_refusal("her session ensure", exc)
        return None
    except Exception:  # noqa: BLE001 — contention/unsupported dir: try next time
        logger.info(
            "librarian: the ensure lock could not be taken; skipping this attempt",
            exc_info=True,
        )
        return None
    hers: str | None = None
    try:
        existing = state.session_id_of(root)
        if existing and (_sessions_root(root) / existing).is_dir():
            _reconcile_session_title(root, existing)
            hers = existing
        else:
            session_id = uuid.uuid4().hex[:12]
            # The id is recorded BEFORE the directory exists: at every instant
            # another scan can run, either the directory does not exist yet
            # (nothing to count) or it is the one the recorded id excludes.
            state.update_state(root, session_id=session_id, paused_at=None)
            try:
                await _create_session_dir(root, session_id)
            except Exception:
                _discard_failed_create(root, session_id)
                # Put the record back: the id above names a directory that
                # does not exist, and leaving it would look like a recorded
                # session until the next attempt overwrites it.
                state.update_state(root, session_id=None, paused_at=None)
                raise
            hers = session_id
    except Exception:  # noqa: BLE001 — boot paths must not fail on her account
        logger.warning("librarian: ensure_session could not create the session", exc_info=True)
        return None
    finally:
        await asyncio.to_thread(lock.release)

    # THE PIN IS REASSERTED ON EVERY ENSURE, not only when the directory was
    # created. QA on #2627 measured the failure that made this load-bearing: a
    # concurrent first boot left the pin file holding only Aida's id, and every
    # later boot took the existing-session branch, so the Librarian was never
    # pinned again — present, titled and armed, but absent from the sidebar
    # forever. ``set_pin`` is desired-state and writes nothing when the id is
    # already there, so the steady-state cost is one lock plus one small read.
    if hers is not None:
        try:
            from local_operator.tui.sidebar_pins import (
                SOURCE_BOOTSTRAP,
                explicitly_unpinned,
                set_pin,
            )

            # A DELIBERATE UNPIN WINS. The reassert exists to repair a pin lost
            # to the concurrent-boot race, not to overrule a person who removed
            # the row; the ledger records only human writes (``set_pin`` and
            # ``toggle_pin``), so a bootstrap reassert can never create or clear
            # one.
            if hers not in explicitly_unpinned(root):
                await asyncio.to_thread(set_pin, root, hers, True, source=SOURCE_BOOTSTRAP)
        except Exception:  # noqa: BLE001 — a pin is decoration; the session is not
            logger.warning("librarian: could not pin her session", exc_info=True)

    # HER CADENCE IS ARMED ON EVERY CALL, NOT ONLY AT CREATION. An incarnation
    # that got a cadence once, at birth, and was never asked again would lose
    # it permanently on any later row loss (an upgrade, a hand-edited index, a
    # crash between an external arm's transcript append and its index write).
    # ``ensure_armed`` is the idempotent question ("is the cadence row there?
    # arm it if not") and refuses a session a live runtime owns.
    if hers is not None:
        await proactive.ensure_armed(root, hers, now_ms=now_ms)

    # THEN REAP WHAT THE PREVIOUS INCARNATION LEFT BEHIND. Scoped to engine
    # rows on sessions with no transcript, never her own entry. Off-thread and
    # best-effort: a boot never fails on her account.
    if hers is not None:
        try:
            await asyncio.to_thread(proactive.reap_orphan_rows, root, keep=hers)
        except Exception:  # noqa: BLE001 — litter, not a boot dependency
            logger.warning("librarian: could not reap orphaned wake rows", exc_info=True)
    return hers
