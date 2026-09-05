"""One-shot config migrations, run from ONE explicit startup seam.

Why a module of its own, and why the seam matters more than the migration:

The first version of the session-cleanup migration lived inside
``ConfigManager._load_config`` and rewrote ``config.yml`` on every load that
found a retired key. That made READING the config a WRITE, so any process
that so much as constructed a ``ConfigManager`` — a reviewer's probe, a
debugging one-liner with the worktree on ``sys.path`` and no ``HOME``
isolation — silently migrated whatever config dir it resolved. One did: an
un-isolated probe migrated the operator's live config while the change was
still under review (PR #645, round 5).

That alone would have been embarrassing. What made it dangerous is the
second half: the migration REMOVED ``session.reap_unused: false``, the key
the *installed* runtime's reaper still read as its opt-out. The installed
``lop`` was an older version, the config no longer guarded it, and the next
idle launch reaped a session with a transcript. On any machine, the window
between "config migrated" and "every process is on the new version" — a
live TUI, the mobile daemon, a wake supervisor, another worktree's venv —
is exactly when the old reaper runs against a config that no longer
protects the user from it.

So, two rules, both enforced here and by tests:

1. **A migration runs from :func:`run_startup_migrations` and nowhere
   else.** ``cli.main`` calls it once, for the config dir the command is
   about to use. Construction of ``ConfigManager`` never triggers it;
   ``settings_io`` never triggers it; the TUI never triggers it. There is
   NO stamp file: the gate is the migration's own "would this change
   anything?" predicate, evaluated against the config every launch. A stamp
   was tried and had three defects for one benefit — a corrupt stamp raised
   on the start path, a failed backup got stamped and left the belt
   unfastened for good, and a config restored from a backup was skipped as
   "done" (review round 5, R5-2/3/4). The benefit was one ``stat`` saved on
   a config ``lop`` is about to read anyway.

2. **Retired opt-out keys are WRITTEN, never removed.** The migration sets
   ``values["session.reap_unused"] = False`` (the flat-dotted key the #576
   reaper read) AND ``values.session.reap_unused = False`` (the nested key
   ``/settings`` wrote), and leaves both in place permanently. Current code
   ignores them; every older runtime that can still start on this machine
   is held off by them. The cost is two inert lines in ``config.yml``; the
   alternative cost was the incident.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Left behind by the one release candidate that stamped (PR #645 round 5,
#: never shipped). Harmless and ignored; named so a reader of a config dir
#: knows what it is. Nothing writes it any more.
LEGACY_STAMP_NAME = ".migrations"

#: The retired ceilings of the first eviction policy. Removed: nothing reads
#: them at any version that also carries this module, and an older runtime
#: treated any value as "retired and ignored" with a warning, so their
#: presence protected nothing.
_RETIRED_CEILINGS = (
    "session_retention_max_sessions",
    "session_retention_max_bytes",
    "session_retention_max_age_days",
)

#: The #576 reaper's opt-out, in the spelling its ``sweep_from_config``
#: actually read (``Config.values.get("session.reap_unused")`` — a flat
#: dotted key). KEPT and pinned to False: see the module docstring.
_REAP_UNUSED_FLAT = "session.reap_unused"


def migrate_session_cleanup(config_dir: Path) -> list[str]:
    """Pin the old reapers OFF and opt the user out of the new cleanup policy.

    Returns the list of changes made (empty when the config already had the
    final shape), so the caller and the tests can see exactly what moved.
    Idempotent: a second run on the migrated file changes nothing and writes
    nothing — and that no-op IS the startup gate, so a config restored from
    a backup (retired keys back, opt-out gone) is migrated again on the next
    launch. Backs ``config.yml`` up beside itself before any rewrite; if the
    backup cannot be written the file is left alone (the keys it would add
    are protective, but a user losing the record of what they had set is the
    worse outcome) and the next launch retries, because nothing records the
    attempt as done.

    Changes, in order:

    * ``values["session.reap_unused"] = False`` and
      ``values.session.reap_unused = False`` — WRITTEN, whatever they were.
      The nested form is what ``/settings`` wrote; the flat form is what the
      old reaper read; both are kept so no older runtime that can still start
      on this machine reaps anything. A value of ``True`` in either spelling
      is overwritten: the user's standing instruction is that nothing removes
      sessions unless explicitly enabled, and the new policy is the only
      explicit switch.
    * ``session_retention_max_*`` removed (inert at every version).
    * ``session.cleanup.enabled = False`` written EXPLICITLY when absent — an
      explicit false survives a future change of default and is visible to
      anyone reading the file. An existing ``cleanup`` block is merged into.
    * The session store is marked (``sessions/.local-operator-store``) so
      that IF the user later enables cleanup the store is eligible. Marking
      enables nothing — ``enabled`` was just pinned to false — and this is
      the only place outside session construction that marks.
    """
    from local_operator.config import ConfigManager

    config_file = config_dir / "config.yml"
    if not config_file.is_file():
        return []
    manager = ConfigManager(config_dir)
    values: dict[str, Any] = manager.get_config().values
    changes: list[str] = []

    if values.get(_REAP_UNUSED_FLAT) is not False:
        values[_REAP_UNUSED_FLAT] = False
        changes.append(f"{_REAP_UNUSED_FLAT} (flat) = false")
    session = values.get("session")
    if not isinstance(session, dict):
        session = {}
        values["session"] = session
    if session.get("reap_unused") is not False:
        session["reap_unused"] = False
        changes.append("session.reap_unused (nested) = false")
    for key in _RETIRED_CEILINGS:
        if key in values:
            del values[key]
            changes.append(f"removed {key}")
    cleanup = session.get("cleanup")
    if not isinstance(cleanup, dict):
        cleanup = {}
        session["cleanup"] = cleanup
    if "enabled" not in cleanup:
        cleanup["enabled"] = False
        changes.append("session.cleanup.enabled = false")
    if not changes:
        return []

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = config_file.with_name(f"{config_file.name}.pre-cleanup-migration.{stamp}")
    try:
        backup.write_bytes(config_file.read_bytes())
    except OSError as exc:
        logger.warning(
            "config migration: could not back up %s (%s); leaving it as is and "
            "retrying at the next launch",
            config_file,
            exc,
        )
        return []
    # ``values`` IS the manager's live dict (mutated in place above, including
    # the deletions), so the write is the manager's own serialisation of it;
    # ``update_config`` would merge key-by-key and could not express a delete.
    # That serialisation is the manager's FULL view: ``_load_config`` filled
    # every absent top-level default in, so the rewritten file carries keys
    # (``compaction``, ``web_fetch``, …) the original omitted. Same values,
    # spelled out — the backup keeps the user's original spelling.
    manager._write_config(vars(manager.config))

    from local_operator.session.cleanup import SESSIONS_DIRNAME, mark_store

    if (config_dir / SESSIONS_DIRNAME).is_dir():
        mark_store(config_dir / SESSIONS_DIRNAME)

    logger.warning(
        "config migration: %s in %s (backup at %s); the retired reapers' opt-out is "
        "pinned off for any older runtime, and no automatic session cleanup runs "
        "unless you turn it on in /settings",
        "; ".join(changes),
        config_file,
        backup,
    )
    return changes


def run_startup_migrations(config_dir: Path) -> None:
    """THE seam. Called once by ``cli.main`` for the config dir it will use.

    Best-effort in the strongest sense: a migration that raises — for ANY
    reason, a corrupt file included — must never stop ``lop`` from starting;
    the config it would have touched is still readable by the code that
    ships with it, or is handled by ``ConfigManager`` the same way it would
    have been moments later. No state is recorded: the migration's own
    no-op path is the gate (see the module docstring).
    """
    try:
        migrate_session_cleanup(config_dir)
    except Exception as exc:  # noqa: BLE001 — never a reason not to start
        # One line at WARNING, the traceback at DEBUG: the usual cause is a
        # config ``ConfigManager`` itself cannot read, and the command about
        # to run reports THAT with its own message; a second traceback here
        # would bury it.
        logger.warning("config migration: session-cleanup migration skipped: %s", exc)
        logger.debug("config migration: traceback", exc_info=True)
