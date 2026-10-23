"""One-shot config migrations, run from ONE explicit startup seam.

THREE STORES, ONE SEAM. The first migration here repairs ``config.yml``; two
more (``agent_profiles.backfill_seed_action_class`` and
``agent_profiles.startup_seed_update_pass``) repair the agent registry, and a
fourth (``projects.migrate_coordination_links``) re-kinds the projects store.
Each exists for the same class of reason - a release introduced a datum (or
shipped a newer starter text) and the rows written before it existed need it.
All are called from :func:`run_startup_migrations` and nowhere else; the
rules below are about the seam, not about which file it happens to write.

The seed-update arm is the one arm that is not strictly a repair: it applies
packaged starter updates to rows the revision ledger proves are unedited and
behind (the #2060 fix), and reports the rest. It keeps this seam's doctrine
anyway - idempotent predicate, best-effort, never a reason not to start - and
the display-only exception it adds (``.seed-notices.json``) is documented at
``agent_profiles._load_seed_notice_state``.

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
   a config ``lop`` is about to read anyway. The same rule holds for the
   registry backfill: its predicate is "is this row still missing the datum
   its starter declares?", which a repaired row answers no.

2. **Retired opt-out keys are WRITTEN, never removed.** The migration sets
   ``values["session.reap_unused"] = False`` (the flat-dotted key the #576
   reaper read) AND ``values.session.reap_unused = False`` (the nested key
   ``/settings`` wrote), and leaves both in place permanently. Current code
   ignores them; every older runtime that can still start on this machine
   is held off by them. The cost is two inert lines in ``config.yml``; the
   alternative cost was the incident.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from local_operator.config_writer import serialized_config_path_write

logger = logging.getLogger(__name__)

#: Left behind by the one release candidate that stamped (PR #645 round 5,
#: never shipped). Harmless and ignored; named so a reader of a config dir
#: knows what it is. Nothing writes it any more.
LEGACY_STAMP_NAME = ".migrations"

#: The subcommand spellings whose OWN promise is "change nothing" and which
#: therefore skip the seed-update arm WHOLE: a startup write under them would
#: break the promise — and ``config edit agents.auto_update.seeds`` racing the
#: pass would pre-apply under the OLD value before the user's choice takes
#: effect (UX round 1, U6b). Canonical spellings are computed by
#: ``cli._seed_sync_command``; keep the two in sync.
_NO_WRITE_COMMANDS = frozenset({"agents sync", "config edit agents.auto_update.seeds"})

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

#: The sidecar that records which setting paths the last launch saw, so the next
#: one can say what went missing between them. Dot-prefixed machinery beside the
#: config, never a setting: nothing reads it as configuration and ``/settings``
#: does not list it.
_KEY_INVENTORY_NAME = ".config-keys.json"


def migrate_session_cleanup(config_dir: Path) -> list[str]:
    """Read-only eligibility first; serialize only a migration that can write.

    Startup reaches this even for refused/read-only commands. Creating a lock
    directory/file before discovering there is no config was itself a mutation.
    The locked helper re-reads the actual file before applying any changes, so
    this cheap preflight does not become a stale whole-config write base.
    """
    from local_operator.config import DEFAULT_CONFIG, read_config_values

    if not (config_dir / "config.yml").is_file():
        return []
    raw_values = read_config_values(config_dir)
    if raw_values is None:
        return []
    # Match the existing loader's TOP-LEVEL backfill without constructing it
    # (construction can move malformed config). Existing nested mappings are
    # not backfilled, exactly as _load_config already specifies.
    values = {**DEFAULT_CONFIG.values, **raw_values}
    session = values.get("session")
    cleanup = session.get("cleanup") if isinstance(session, dict) else None
    needs_write = (
        values.get(_REAP_UNUSED_FLAT) is not False
        or not isinstance(session, dict)
        or session.get("reap_unused") is not False
        or not isinstance(cleanup, dict)
        or "enabled" not in cleanup
        or any(key in values for key in _RETIRED_CEILINGS)
    )
    if not needs_write:
        return []
    return _migrate_session_cleanup_locked(config_dir)


@serialized_config_path_write
def _migrate_session_cleanup_locked(config_dir: Path) -> list[str]:
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
    retired: list[str] = []
    for key in _RETIRED_CEILINGS:
        if key in values:
            del values[key]
            retired.append(key)
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
    #
    # The retirements are DECLARED, and only for the keys this run actually
    # removed: ``_write_config`` carries every disk value the payload does not
    # model back through, so without the declaration the three keys the loop above
    # just deleted would be read off disk and rewritten — a retirement that reports
    # success and changes nothing.
    manager._write_config(
        vars(manager.config),
        drop={
            (key,): "retired session-retention ceiling; no reader at any shipped version"
            for key in retired
        },
    )

    from local_operator.session.cleanup import SESSIONS_DIRNAME, mark_store

    if (config_dir / SESSIONS_DIRNAME).is_dir():
        mark_store(config_dir / SESSIONS_DIRNAME)

    # THE LOG LEVEL, AND ONLY THE LOG LEVEL. Under a per-run HOME (``lop
    # exec``, agent-runtime-svc) this migration fires on EVERY run — the run
    # config is rewritten per Execute, so the keys are absent every time — and
    # the WARNING became one stderr line per run, which the runtime-svc adapter
    # persists as an audit event. The writes and the ``.pre-cleanup-migration``
    # backup stay EXACTLY as they were; the module docstring's two rules are
    # untouched. A redirected HOME gets DEBUG; "cannot tell" keeps the
    # WARNING, and so does any failure in the probe — fail loud is the safe
    # direction for the one person who might genuinely want to read it.
    try:
        from local_operator.supervisors import home_is_the_users

        redirected = home_is_the_users() is False
    except Exception:  # noqa: BLE001 — a log level must never fail a migration
        redirected = False
    emit = logger.debug if redirected else logger.warning
    emit(
        "config migration: %s in %s (backup at %s); the retired reapers' opt-out is "
        "pinned off for any older runtime, and no automatic session cleanup runs "
        "unless you turn it on in /settings",
        "; ".join(changes),
        config_file,
        backup,
    )
    return changes


def _read_key_inventory(path: Path) -> "list[str] | None":
    """The recorded setting paths, or ``None`` when there is no usable baseline.

    ``None`` is "no baseline", not "nothing was lost": a missing sidecar, one that
    cannot be read, a ``schema`` this build does not know, or a ``keys`` list that
    is not all strings. A report built on a corrupt sidecar would name settings
    that may never have existed, so the arm re-baselines instead — the same reason
    a restore from backup is migrated again rather than skipped (module docstring).
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict) or raw.get("schema") != 1:
        return None
    keys = raw.get("keys")
    if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
        return None
    return keys


def _write_key_inventory(path: Path, keys: "list[str]") -> None:
    """Record ``keys`` atomically, 0600, best-effort.

    Temp file in the SAME directory plus ``os.replace``, for the reason
    ``ConfigManager._write_config`` documents at length: a crash between a
    truncate and a flush would leave a half-written sidecar, and the next launch
    would read that as "no baseline" and silently re-baseline over the very loss
    this file exists to remember. 0600 because the paths are as private as the
    config they describe (they name provider endpoints, hub hosts, team sessions).

    Best-effort on ``OSError``: a read-only config root or a full disk must not
    stop ``lop`` from starting, and this module's doctrine is that every arm is.
    """
    payload = {
        "schema": 1,
        "keys": sorted(keys),
        "recorded_at": datetime.now().isoformat(),
    }
    handle: "int | None" = None
    temp_path: "Path | None" = None
    try:
        handle, raw_path = tempfile.mkstemp(
            dir=str(path.parent), prefix=".config-keys.", suffix=".tmp"
        )
        temp_path = Path(raw_path)
        os.chmod(raw_path, 0o600)
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            handle = None
            json.dump(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(raw_path, path)
        temp_path = None
    except OSError:
        logger.debug("config key inventory: could not record %s", path, exc_info=True)
    finally:
        if handle is not None:
            try:
                os.close(handle)
            except OSError:
                pass
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def report_config_key_shrink(config_dir: Path, *, surface: str = "cli") -> "list[str]":
    """Say what the last launch's config had and this one does not.

    THE INCIDENT THIS EXISTS FOR (2026-10-23). A QA rig wrote 34 bytes over the
    operator's 26-key ``config.yml`` at 04:21:42 and the next launch expanded it
    to the shipped defaults, so the file afterwards looked like a config somebody
    had freshly created — no error, no diff, and the nine top-level keys and every
    nested user value gone. Nothing in the product could have told them: the loss
    happened BEFORE this build's writers ran. A per-launch leaf-path inventory is
    the only place that remembers.

    LEAF paths, not top-level keys, because the migration then back-fills the
    top-level defaults (``retry``, ``subagents``, …): a top-level inventory would
    see them return and report nothing, while ``retry.fallbackChains.default`` and
    ``subagents.models.hi`` stayed gone.

    THREE DECISIONS, each of which has an incident behind it:

    * A 0-BYTE file is unreadable-as-absent, not "all keys lost". A truncation
      (an interrupted write, a ``> config.yml``) would otherwise report the whole
      file as lost AND, worse, re-baseline to nothing — erasing the record of what
      the file used to hold. Only a PARSED, NON-EMPTY ``values`` mapping rewrites
      the inventory.
    * The inventory is rewritten ONLY when the key set differs, so an unchanged
      read-only launch (``lop config list``, ``--version``, the TUI's boot) puts
      no write on the disk. "Reading is writing" is the defect the whole migration
      seam was moved out of ``_load_config`` to end.
    * ``surface="daemon"`` logs at DEBUG and leaves the inventory ALONE. A daemon
      launch is nobody's answer to "did I lose my settings?", and re-baselining
      there would make the loss permanent before the operator ever saw it — the
      next human launch must still be able to report it.

    ``config.yml`` is never touched here, and neither is a config write lock: two
    launches racing re-baseline atomically and the later one wins, which is the
    same answer either would have reached a moment apart. Returns the missing
    paths, so a caller (and the tests) can assert on what was reported without
    parsing the message.
    """
    from local_operator.config import _dotted, _leaf_paths, read_config_values

    config_file = config_dir / "config.yml"
    try:
        if not config_file.is_file() or config_file.stat().st_size == 0:
            return []
    except OSError:
        return []
    values = read_config_values(config_dir)
    if values is None:
        # Unreadable right now (a torn save, a permissions problem): keep the
        # baseline rather than treating a bad read as "everything unset".
        return []
    # DOTTED STRINGS, the spelling the message uses and the sidecar stores: a JSON
    # list of path tuples would read back as lists (`[["retry", "maxRetries"]]`),
    # which compares equal to nothing and renders as nested arrays in the log.
    current = sorted(_dotted(path) for path in _leaf_paths(values))
    current_set = set(current)
    inventory_path = config_dir / _KEY_INVENTORY_NAME
    previous = _read_key_inventory(inventory_path)
    if previous is None:
        if current:
            _write_key_inventory(inventory_path, current)
        return []
    missing = sorted(set(previous) - current_set - set(_RETIRED_CEILINGS) - {_REAP_UNUSED_FLAT})
    if missing:
        message = (
            f"{config_file} lost {len(missing)} setting(s) since the last launch: "
            f"{', '.join(missing)}; a writer outside this build removed them — "
            "restore from a config.yml.bak-* backup if you did not."
        )
        if surface == "daemon":
            logger.debug("%s", message)
            return missing
        logger.warning("%s", message)
        if surface in ("tui", "cli"):
            print(message, file=sys.stderr)
        return missing
    if current and current_set != set(previous):
        # Additions (the cleanup migration's own keys, a new release's defaults)
        # re-baseline silently; only the missing set is ever reported.
        _write_key_inventory(inventory_path, current)
    return []


def run_startup_migrations(
    config_dir: Path, *, surface: str = "cli", command: str | None = None
) -> None:
    """THE seam. Called once by ``cli.main`` for the config dir it will use.

    Best-effort in the strongest sense: a migration that raises — for ANY
    reason, a corrupt file included — must never stop ``lop`` from starting;
    the config it would have touched is still readable by the code that
    ships with it, or is handled by ``ConfigManager`` the same way it would
    have been moments later. No state is recorded: the migration's own
    no-op path is the gate (see the module docstring).

    Four arms: the session-cleanup config migration, the action-class
    backfill for the agent registry, the starter-update pass (``#2060`` —
    reports drift and auto-applies the rows the ledger proves unedited), and
    the projects coordination re-kind (schema 1 -> 2 — the store-side
    migration lives in
    :func:`local_operator.projects.migrate_coordination_links` and shares the
    same doctrine: idempotent predicate, backup-first, abort-if-no-backup).
    Each arm fails on its own; one skipping never skips the others.

    ``surface``/``command`` describe the invocation for the seed-update arm
    only. ``surface`` is "tui", "cli" or "daemon" and picks the notice
    channel (the TUI queues lines for its boot hook; the CLI prints them
    plainly to stderr; a daemon launch is REPORT-ONLY — it writes no row,
    records nothing and logs at DEBUG, so the first human surface applies
    and announces) —
    ONE definition of "which surface is this", computed by
    ``cli._startup_surface`` and shared with the ``use_tui`` decision so the
    two cannot drift. ``command`` is the subcommand spelling (e.g. "agents
    sync") and SKIPS the arm entirely for the commands that do their own
    read-only/apply work or that change the pass's own switch: a startup
    write under ``lop agents sync --check`` would break the "change
    nothing" promise the command makes and race the state it is checking,
    and ``config edit agents.auto_update.seeds`` must land the new value
    before the next pass reads it. See ``_NO_WRITE_COMMANDS``; nothing else
    in the tree promises no-write, so nothing else is carved out.
    """
    try:
        # FIRST, and before the cleanup migration: that migration back-fills every
        # absent top-level default (``_load_config`` does), so a shrink reported
        # after it would see those keys present and name none of them.
        report_config_key_shrink(config_dir, surface=surface)
    except Exception as exc:  # noqa: BLE001 — never a reason not to start
        logger.warning("config migration: key inventory check skipped: %s", exc)
        logger.debug("config migration: traceback", exc_info=True)
    try:
        migrate_session_cleanup(config_dir)
    except Exception as exc:  # noqa: BLE001 — never a reason not to start
        # One line at WARNING, the traceback at DEBUG: the usual cause is a
        # config ``ConfigManager`` itself cannot read, and the command about
        # to run reports THAT with its own message; a second traceback here
        # would bury it.
        logger.warning("config migration: session-cleanup migration skipped: %s", exc)
        logger.debug("config migration: traceback", exc_info=True)
    try:
        # THE SECOND STORE THIS SEAM MIGRATES, and it is here for the same
        # reason as the first: a release that introduces a datum must repair the
        # rows written before it existed, and there is exactly one place a
        # release may do that from. This one writes to the AGENT REGISTRY (rows
        # installed from a packaged starter before the starter declared a
        # class), not to ``config.yml`` — see ``backfill_seed_action_class``
        # for the predicate and the failure it exists to end: an install whose
        # Aida went permanently silent on upgrade because an absent class tag
        # reads as "reactive" (the operator's 2026-09-30 report).
        from local_operator.agent_profiles import backfill_seed_action_class

        backfill_seed_action_class(config_dir)
    except Exception as exc:  # noqa: BLE001 — never a reason not to start
        logger.warning("config migration: action-class backfill skipped: %s", exc)
        logger.debug("config migration: traceback", exc_info=True)
    if command not in _NO_WRITE_COMMANDS:
        try:
            # THE THIRD AGENT-REGISTRY ARM, and the #2060 fix proper: the
            # upgrade that brings a newer packaged starter should bring its
            # text too, reported or applied per the ledger's proof. Imported
            # lazily - this module must not drag the registry (dill, yaml)
            # onto every CLI start just because one arm may touch it - and
            # skipped WHOLE for the commands whose own body does the work (see
            # ``_NO_WRITE_COMMANDS``).
            from local_operator.agent_profiles import startup_seed_update_pass

            startup_seed_update_pass(config_dir, surface=surface)
        except Exception as exc:  # noqa: BLE001 — never a reason not to start
            logger.warning("config migration: seed update pass skipped: %s", exc)
            logger.debug("config migration: traceback", exc_info=True)
    try:
        # Lazily imported: the store module (pydantic models, the runtime
        # scan's dependencies) must not ride the import path of every CLI
        # start just because one migration may touch it.
        from local_operator.projects import migrate_coordination_links

        migrate_coordination_links(config_dir)
    except Exception as exc:  # noqa: BLE001 — never a reason not to start
        logger.warning("config migration: project coordination migration skipped: %s", exc)
        logger.debug("config migration: traceback", exc_info=True)
