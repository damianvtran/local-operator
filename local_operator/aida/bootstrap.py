"""Aida's bootstrap: ensure her single session exists, pinned, and armed.

R7's mechanics. :func:`ensure_session` is the ONE place her session directory
is created, called from the TUI boot, the server lifespan, ``/aida`` and the
desktop ``open``/``greet`` ops — every entry point routes through it, so two
concurrent first invocations cannot mint two conversations (``ensure.lock``
serialises them and the state file is re-read under it; the loser returns the
id the winner wrote).

HOW THE SESSION IS CREATED, and why not by constructing one. The design named
``session_factory.create_session``; constructing a Session needs a resolvable
provider+model, and does real work (MCP wiring, skills, classification) an
empty conversation has no use for — and R28 requires ``/aida`` to open (and
this bootstrap to run) on an install with NO provider configured, where the
factory raises ``HostingNotConfiguredError``. So the bootstrap writes the
durable facts a session directory is made of, exactly as the desktop's own
``create`` writes a bare directory plus a marker:

1. mint a session id (``uuid4().hex[:12]`` — the shape every other mint uses;
   ``pathlib``-safe by construction, so it needs no name validation);
2. ``mkdir(sessions/<id>, mode=0o700)`` — ``exist_ok=False`` under the lock: a
   collision means the id was not fresh and must retry loudly, not silently
   adopt a directory nobody checked;
3. ``ensure_session_created_at`` — the same created-at sidecar the factory's
   construction writes, so the picker's two clocks work on her row;
4. ``write_session_attachment(agent="aida")`` — the sidecar the session load
   path restores as her attached role (``Session._load_attachment`` →
   ``attach_agent_profile``). The packaged seed resolves through
   ``resolve_profile`` even uninstalled, so nothing user-authored is required
   (R1) and ``/agent aida`` works for free (§2.1);
5. the conversation title, both channels: the ``conversation_name`` custom
   entry (the transcript reader's fallback) and the title sidecar (the O(1)
   picker read), ``user_set=True`` so the auto-namer may not replace it after
   her first turn — the configured ``aida.name`` (default "Aida") is what
   R27's row must say;
6. a birth custom entry, which is ALSO the reason the directory becomes REAL:
   ``session_activity`` (the ONE ranking clock) returns ``None`` for a
   directory holding neither a transcript nor a mail spool, and such a
   directory is neither listed nor resumable (``resume_dir`` refuses it) —
   precisely the state ``/aida``'s ensure→resume handoff and the picker row
   must not be in;
7. write ``state.json`` — AFTER the directory exists, so a crash in between
   leaves an orphan directory (invisible: no transcript) rather than a state
   file pointing at nothing;
8. pin her (``sidebar_pins.set_pin``, desired-state, idempotent) so the
   sidebar's pinned-first ordering (R27) holds from the first frame;
9. arm the cadence through :func:`local_operator.aida.proactive.ensure_armed`
   — the transcript-first external writer, which also installs the wake
   supervisor (R10's always-on property: she fires with every terminal
   closed). Steps 1-8 build the session; THIS step runs on EVERY call, for an
   existing id as well as a new one, because "she has a cadence" is not a fact
   a creation can establish once and forget: any later loss of the row (see
   ``ensure_armed``'s return words, and ``reap_orphan_rows`` below it) used to
   be permanent and silent. Releasing the lock first is what makes that
   possible — the armer takes its own;
10. reap the rows the PREVIOUS incarnation left on the wake index
    (:func:`local_operator.aida.proactive.reap_orphan_rows`) — engine rows
    parked on a session id with no transcript, which no runtime can ever be
    started for and which the operator otherwise clears by hand.

STEP 9 IS WHY THE RE-CREATION PATH IS NOT AN EARLY RETURN any more: the old
shape returned ``existing`` from inside the lock, so an install whose class tag
an upgrade had left unwritten (the live cause on 2026-09-30) never asked the
question again. Steps 9 and 10 are best-effort by contract, like every other
side effect here.

THE LOCK IS HELD ACROSS THE CREATE, on the documented wake-lock pattern: the
acquire/release hop to a worker thread so the event loop stays free, and the
work between them is a handful of file writes (no model, no provider). A
contended lock is answered with ``None`` and a log line — every caller is
best-effort by design, and the next boot retries.

DISABLED. The gate runs first — before the lock, before any path is joined —
and answers ``None`` with zero writes (R17's zero-footprint promise, pinned by
``tests/unit/aida/test_aida_bootstrap.py``'s fs-snapshot test).
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
import uuid
from pathlib import Path

from local_operator.aida import naming, proactive, state
from local_operator.wakes.lock import WakeLockBusy, WakeLockUnavailable

logger = logging.getLogger(__name__)

#: The master switch's default. Defined by the engine (its reader) and
#: re-exported here so the boot gate and the engine share one value; the
#: settings anti-drift test reads it from here for the ``aida.enabled`` row.
DEFAULT_ENABLED = proactive.DEFAULT_ENABLED

#: Her packaged role's name: the seed file (``agent_seeds/aida.md``), the
#: attachment sidecar's ``agent`` field, and the name ``/agent aida`` resolves.
ROLE_NAME = "aida"

#: Her conversation's packaged title — the default the configured name
#: starts from. What creation actually writes is ``aida.name`` (see
#: :mod:`local_operator.aida.naming`), reconciled on every ensure; this
#: constant stays as that default and for the module's existing readers.
SESSION_TITLE = naming.DEFAULT_NAME

#: The birth custom entry's type. Never enters the model's context; it is the
#: activity clock's reason to rank her row and a permanent marker of how the
#: conversation was born.
BIRTH_CUSTOM_TYPE = "aida_session"


def config_enabled(config_dir: Path | str | None = None) -> bool:
    """Whether Aida is enabled: the config key AND the environment switch.

    ``aida.enabled`` (default true) is read through a fresh ``ConfigManager``
    with the same never-raise posture every boot-path config read uses; the
    env switch is :func:`local_operator.aida.state.env_disabled`. Either one
    disables. Import-light: ``local_operator.config`` is imported lazily.
    """
    if state.env_disabled():
        return False
    try:
        from local_operator.config import ConfigManager
        from local_operator.paths import config_dir as resolve_config_dir

        root = Path(config_dir) if config_dir is not None else resolve_config_dir()
        raw = ConfigManager(config_dir=root).get_nested_value(("aida", "enabled"), DEFAULT_ENABLED)
    except Exception:  # noqa: BLE001 — an unreadable config must not enable-or-crash
        logger.warning("aida: could not read aida.enabled; treating as enabled")
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
    # AFTER the operator renamed her (``aida.name``, a /settings edit) must be
    # born wearing that name — the sidebar's pinned row and the picker read
    # this title, and they cannot ask a session that does not exist yet.
    title = naming.display_name(config_dir)
    ensure_session_created_at(directory, time.time())
    write_session_attachment(directory, team="", agent=ROLE_NAME, goal="")
    write_session_title(directory, title, user_set=True, past_names=[])
    transcript = Transcript(directory)
    # The name entry rides BEFORE the birth entry so the transcript reader's
    # first window already holds the title if the sidecar was lost.
    entry = {
        "text": title,
        "user_set": True,
    }
    from local_operator.session.naming import CONVERSATION_NAME_CUSTOM_TYPE

    await transcript.append_custom(CONVERSATION_NAME_CUSTOM_TYPE, entry)
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
        logger.debug("aida: could not clean up a failed session create", exc_info=True)


async def ensure_session(
    config_dir: Path | str | None = None, *, now_ms: int | None = None
) -> str | None:
    """Her session id, creating the session on first need. ``None`` when disabled.

    Idempotent on every call after the first, and deliberately not free: one
    state read, one directory stat, one title check (which reconciles her
    stored title to ``aida.name`` — see ``local_operator.aida.naming``), one
    wake-index read to answer "is her cadence armed?" and one listing of the
    index to reap any orphaned engine rows (step 9/10 above). What it never
    does is construct a session or resolve a provider — the two costs that
    would make a boot hook expensive. Safe to call from boot paths
    (best-effort: any failure logs and answers ``None``) and from every
    ``/aida``/desktop op.
    """
    from local_operator.paths import config_dir as resolve_config_dir

    root = Path(config_dir) if config_dir is not None else resolve_config_dir()
    if not config_enabled(root):
        return None

    # PUBLISH THE TRIGGER SETTINGS SNAPSHOT AT BOOT (the design's writers (c)):
    # the wake supervisor may evaluate long before her first reconcile, and the
    # snapshot must describe THIS config, not a stale one. Best-effort by
    # contract — a boot never fails on its account — and cheap when the values
    # did not move (the publish then writes nothing).
    try:
        from local_operator.aida import proactive as _proactive

        _proactive.publish_trigger_settings(root)
    except Exception:  # noqa: BLE001 — a snapshot never fails a boot
        logger.warning("aida: could not publish the trigger settings snapshot", exc_info=True)

    lock = state.wake_lock(root)
    try:
        await asyncio.to_thread(lock.acquire)
    except (WakeLockBusy, WakeLockUnavailable) as exc:
        # A refusal, not a defect: the shared shape (:func:`state.note_lock_refusal`)
        # is what keeps a stack out of the operator's log. The line used to be an
        # INFO WITH ``exc_info`` — quiet to read, and still a `Traceback` block in
        # the file the e2e clean-log contract greps, which is the residual the
        # reviewer measured on a held lock. The next ensure tries again.
        state.note_lock_refusal("her session ensure", exc)
        return None
    except Exception:  # noqa: BLE001 — contention/unsupported dir: try next time
        logger.info(
            "aida: the ensure lock could not be taken; skipping this attempt", exc_info=True
        )
        return None
    created: str | None = None
    hers: str | None = None
    try:
        existing = state.session_id_of(root)
        if existing and (_sessions_root(root) / existing).is_dir():
            # CONFIG IS CANONICAL FOR HER NAME: a rename made while she has no
            # open runtime (a /settings edit, /aida rename in a terminal that
            # is not sitting on her conversation, the desktop while she is
            # closed) must reach the picker and the sidebar, which read the
            # stored title from disk — not only the live session's watcher.
            # Cheap when nothing moved: one small sidecar read, compare, done.
            await naming.reconcile_session_title(root, existing)
            hers = existing
        else:
            session_id = uuid.uuid4().hex[:12]
            # HER ID IS RECORDED BEFORE HER DIRECTORY EXISTS, and that order is
            # load-bearing rather than tidy. Two boot hooks run concurrently: this
            # one, and the TUI's ``_route_first_run_boot`` — which asks
            # ``cadence_allowed`` whether the install already has conversations,
            # and whose verdict SETTLES the ledger (``owed`` -> ``skipped``,
            # permanently; see its docstring). ``other_user_sessions`` excludes
            # the one session ``aida/state.json`` names, so in the window where
            # the directory existed and the id did not, a scan excluded NOTHING
            # and counted her own brand-new session as the operator's — its
            # born journal (the title and birth custom rows written just below)
            # is non-empty, which is all ``_counts_as_operator_conversation``
            # needs. CI ``tui-e2e`` on ubuntu-latest lost that race every run
            # (2026-10-09): a fresh isolated root came out of its first boot with
            # ``greeting: {state: skipped}`` stamped in the same millisecond as
            # ``state.created_at``, so she never introduced herself and the
            # cadence armed instead — the failure the leg's diagnosis dump
            # pinned. Recording the id first removes the window: at every
            # instant a scan can run, either her directory does not exist yet
            # (nothing to count) or it is the one her recorded id excludes.
            state.update_state(root, session_id=session_id, paused_at=None)
            try:
                await _create_session_dir(root, session_id)
            except Exception:
                _discard_failed_create(root, session_id)
                # Put the record back: the id above names a directory that does
                # not exist, and leaving it would exclude nothing while looking
                # like a recorded session until the next attempt overwrites it.
                state.update_state(root, session_id=None, paused_at=None)
                raise
            created = session_id
            hers = session_id
    except Exception:  # noqa: BLE001 — boot paths must not fail on her account
        logger.warning("aida: ensure_session could not create the session", exc_info=True)
        return None
    finally:
        await asyncio.to_thread(lock.release)

    if created is not None:
        try:
            from local_operator.tui.sidebar_pins import set_pin

            await asyncio.to_thread(set_pin, root, created, True)
        except Exception:  # noqa: BLE001 — a pin is decoration; the session is not
            logger.warning("aida: could not pin her session", exc_info=True)

    # HER CADENCE IS ARMED ON EVERY BOOT, NOT ONLY AT HER CREATION, and that one
    # word is the whole of the reported defect's shape. This call used to sit
    # under the create branch alone: an incarnation got its cadence once, at
    # birth, and nothing ever asked the question again. So any later loss — the
    # class tag an upgrade left unwritten (the live cause on 2026-09-30),
    # a filtered drop at load, a hand-edited index, a crash between an external
    # arm's transcript append and its index write — was permanent and silent,
    # and the operator's requirement is the opposite: *she never drops her
    # check-in unless she is asked to*. ``ensure_armed`` is the idempotent
    # question ("is the cadence row there? arm it if not"), it refuses a
    # session a live runtime owns (that session's own reconcile is the writer),
    # and it is the SAME call the create branch uses — one armer, two entry
    # points, so a boot and a re-creation cannot disagree about what arming
    # means.
    if hers is not None:
        await proactive.ensure_armed(root, hers, now_ms=now_ms)

    # THEN REAP WHAT THE PREVIOUS INCARNATION LEFT BEHIND. A re-creation mints
    # a new id, and the old id's armed rows are rows no runtime can ever be
    # started for; the operator had to clear three by hand. Scoped to engine
    # rows on sessions with no transcript, never her own entry — see
    # ``proactive.reap_orphan_rows``. Off-thread and best-effort: a boot never
    # fails on her account, and the sweep is a listing plus a stat per entry.
    if hers is not None:
        try:
            await asyncio.to_thread(proactive.reap_orphan_rows, root, keep=hers)
        except Exception:  # noqa: BLE001 — litter, not a boot dependency
            logger.warning("aida: could not reap orphaned wake rows", exc_info=True)
    return hers
