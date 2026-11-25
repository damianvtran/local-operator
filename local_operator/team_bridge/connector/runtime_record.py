"""Which receiver build is ACTUALLY running — recorded by it, read by ``status``.

WHY THIS EXISTS (remote rollout confirmation). A machine can run ``lop`` at one
version while the always-on receiver, started earlier by the OS service manager
(launchd ``KeepAlive``, a systemd user unit), keeps serving the code it was
exec'd from: an update moves the installed files, never a process that already
imported them. Measured on a teammate's machine at lop v0.102.4 with the
receiver loaded under launchd: nobody could tell which code generation it was,
so nobody could confirm whether the consent-v3 client (``consent_versions``)
had reached that machine's receiver. ``lop team-bridge receiver status`` reads
this record to answer "which version, which build, which consent set, since
when, which pid", and says ``restart needed`` when that is not the installed
version — a question an owner or an agent can now answer over a pasted status.

WHAT IS RECORDED. The process's own facts and nothing else: pid, the package
version it loaded (``update.installed_version`` read IN the receiver, so a
generation install reports that generation's metadata and an editable checkout
its ``pyproject.toml`` at start), the generation directory name when its argv
names one (``update.generation_of_process``), the consent versions this code
treats as default-on (``DEFAULT_ON_CONSENT_VERSIONS`` — the set a hub-served
consent must belong to before this build acts on it), its wall-clock start, and
whether it holds the machine-wide singleton yet. No account, tenant, endpoint
or credential identifier: the status line is meant to be pasted into a chat.

ONE FILE, NEVER CLOBBERING A LIVE PEER. The connector singleton
(:mod:`.lock`) is taken only once an owner exists, so a receiver on a machine
that is not enrolled idles WITHOUT the lock, and a second receiver can start
while the first still holds it (the stale-generation defect :mod:`.reap`
answers). The write rules keep the record naming the process that matters:

* at start, write only when the existing record's pid is not a live receiver
  (or is ours) — a newer process refused by a frozen older holder must not
  hide that holder, which is exactly the receiver ``status`` has to name;
* after the singleton is acquired, always write (``singleton: held``): that
  process is now the one serving this config dir, and it re-asserts that
  record while it holds the lock (:func:`keep_own_record`), because a starter
  that lost the start race can overwrite it and then remove its own;
* on a clean exit, remove the record only when it still names our pid.

A SIGKILL'd receiver leaves its record behind; the reader checks liveness (and,
where ``ps`` answers, that the pid is still a receiver for THIS config dir, so a
reused pid is not mistaken for one) and reports "has exited" rather than a
running build. A receiver started by a version older than this module writes
no record at all; ``status`` pairs a missing (or exited) record with the
service manager's ``loaded`` state to say a restart is needed (``no_record``).

SHAPE. Small JSON beside the bridge store, atomic replace, the same disposable
side-file pattern as ``wake_diagnostics.json``: not a ``state.sqlite3`` table,
because that schema is migration-versioned and this state is worthless once the
process is gone. NEVER RAISES on the writer side — a receiver must not fail to
deliver because a diagnostic could not be written. The reader never creates,
repairs or removes the file (``status`` runs on machines never enrolled).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from local_operator.atomic_files import atomic_write_text

logger = logging.getLogger(__name__)

_SCHEMA = 1

#: ``singleton`` values. ``waiting`` covers both "no owner yet" (idle, the lock
#: is taken only once an owner exists) and "another process holds the lock".
SINGLETON_WAITING = "waiting"
SINGLETON_HELD = "held"


def record_path(config_dir: Path | str) -> Path:
    return Path(config_dir) / "team_bridge" / "connector_runtime.json"


def _read(path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("pid"), int):
        return None
    return raw


def _own_facts(*, started_at: float, singleton: str) -> dict[str, Any]:
    """This process's record. Every probe degrades to ``""``/``None``, never raises."""

    from local_operator.team_bridge.consent_versions import DEFAULT_ON_CONSENT_VERSIONS

    try:
        from local_operator.update import installed_version

        version = installed_version()
    except Exception:  # noqa: BLE001 — a diagnostic never fails the receiver
        version = ""
    try:
        from local_operator.update import generation_of_process

        generation_dir = generation_of_process(os.getpid())
        generation = generation_dir.name if generation_dir is not None else None
    except Exception:  # noqa: BLE001
        generation = None
    return {
        "schema": _SCHEMA,
        "pid": os.getpid(),
        "version": version,
        "generation": generation,
        "consent_versions": sorted(DEFAULT_ON_CONSENT_VERSIONS),
        "started_at": started_at,
        "singleton": singleton,
        "recorded_at": time.time(),
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(payload, sort_keys=True))
    except OSError as error:
        # The writer has already removed its own temp.
        logger.warning("bridge connector runtime record not written: %s", type(error).__name__)


def record_start(config_dir: Path | str, *, started_at: float) -> None:
    """At process start: record ourselves unless a live receiver already is. Never raises.

    Forks ``ps`` (generation and peer liveness) once per process start, before
    any network call — the same one-shot cost the start path already pays.

    The facts are computed BEFORE the existing record is read (review m1): the
    ``ps`` fork in :func:`_own_facts` takes tens of ms, and reading first left
    that whole window between "no live peer" and our write, long enough for a
    concurrent starter to take the singleton, write ``held``, and be
    overwritten by us. Read-then-replace is now microseconds apart; the
    holder's :func:`keep_own_record` closes what remains.
    """

    path = record_path(config_dir)
    try:
        facts = _own_facts(started_at=started_at, singleton=SINGLETON_WAITING)
        existing = _read(path)
        if existing is not None and existing["pid"] != os.getpid():
            if _receiver_alive(existing["pid"], config_dir) is not False:
                return
        _write(path, facts)
    except Exception as error:  # noqa: BLE001
        logger.warning("bridge connector runtime record skipped: %s", type(error).__name__)


def record_singleton_held(config_dir: Path | str, *, started_at: float) -> dict[str, Any] | None:
    """After the singleton is acquired: this process now serves the config dir.

    Returns the facts written so :func:`keep_own_record` can re-assert them
    without forking ``ps`` again; ``None`` when they could not be computed.
    """

    try:
        facts = _own_facts(started_at=started_at, singleton=SINGLETON_HELD)
        _write(record_path(config_dir), facts)
        return facts
    except Exception as error:  # noqa: BLE001
        logger.warning("bridge connector runtime record skipped: %s", type(error).__name__)
        return None


def ensure_own_record(config_dir: Path | str, facts: dict[str, Any]) -> bool:
    """Holder only: rewrite ``facts`` when the record is missing or names another pid.

    The lock makes this process THE receiver for the config dir, so a missing
    or foreign record is always wrong here: a starter that lost the race
    (review m1) overwrote ours and then removed its own on exit. Returns
    whether it rewrote. One small file read when nothing is wrong; never raises.
    """

    path = record_path(config_dir)
    try:
        existing = _read(path)
        if existing is not None and existing["pid"] == os.getpid():
            return False
        _write(path, {**facts, "recorded_at": time.time()})
        return True
    except Exception:  # noqa: BLE001
        return False


#: How often the holder re-checks its record. Not a timeout: it bounds only how
#: long ``status`` can show "no details recorded" after the start race above,
#: a human diagnostic, against one ~200-byte read per period.
KEEP_RECORD_INTERVAL_S = 30.0


async def keep_own_record(
    config_dir: Path | str, facts: dict[str, Any], *, interval: float = KEEP_RECORD_INTERVAL_S
) -> None:
    """Holder's background task: :func:`ensure_own_record` every ``interval`` until cancelled."""

    import asyncio

    while True:
        await asyncio.sleep(interval)
        ensure_own_record(config_dir, facts)


def clear_own_record(config_dir: Path | str) -> None:
    """On a clean exit: remove the record only if it still names this pid."""

    path = record_path(config_dir)
    try:
        existing = _read(path)
        if existing is not None and existing["pid"] == os.getpid():
            path.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass


def _serves(argv: str, config_dir: Path | str) -> bool | None:
    """Does a receiver's ``ps`` argv serve THIS config dir? ``None`` = cannot tell.

    Review m4(b): a pid reused by a receiver for ANOTHER ``--config-dir`` must
    not be reported with this record's version. ``--config-dir`` is a required
    argument of the receiver, so the argv always names it; ``ps`` joins argv
    with spaces, so a config path containing a space cannot be split back out
    reliably — that case answers "cannot tell" rather than a false "gone".
    """

    from local_operator.team_bridge.connector import reap

    root = Path(config_dir)
    spellings = {str(root)}
    try:
        spellings.add(str(root.resolve()))
    except (OSError, ValueError):
        pass
    padded = f"{argv} "
    if any(f"--config-dir {s} " in padded or f"--config-dir={s} " in padded for s in spellings):
        return True
    if reap._same_config(argv, root):
        return True
    if any(" " in s for s in spellings):
        return None
    return False


def _receiver_alive(pid: int, config_dir: Path | str) -> bool | None:
    """``True`` proven live receiver for ``config_dir``, ``False`` gone, ``None`` unsure.

    ``ps`` confirms the pid is still running the receiver module for this
    config dir, so a pid the kernel handed to an unrelated process (or to a
    receiver serving another store) after a SIGKILL is not reported as this
    record's running build. Where ``ps`` cannot answer (Windows, a sandbox) a
    live pid is UNPROVEN — ``None``, never ``True``: the running line still
    shows, but no restart verdict is claimed for it (review m4(a)).
    """

    from local_operator.procstate import pid_liveness

    if pid <= 0:
        return False
    alive = pid_liveness(pid)
    if alive is False:
        return False
    from local_operator.team_bridge.connector import reap

    if not reap._is_posix():
        return None
    process = reap._read_process(pid)
    if process is None:
        return None
    if not reap._is_connector(process.argv):
        return False
    serves = _serves(process.argv, config_dir)
    if serves is False:
        return False
    return True if serves and alive is True else None


def running_receiver(config_dir: Path | str) -> dict[str, Any] | None:
    """The recorded receiver, annotated for ``status``; ``None`` when none recorded.

    Adds ``alive`` (``True``/``False``/``None`` = could not tell),
    ``installed_version`` (this reading process's — i.e. the installed ``lop``'s
    — version), ``generation_stale`` (``update.generation_staleness_of_process``,
    tri-state, only for a live pid), ``restart_reason`` and ``restart_needed``.
    A cause is ``version_mismatch`` (the recorded version differs from a known
    installed version) or else ``replaced_build`` (same version, but the
    generation is provably not ``current`` — an update replaced the files under
    it). ``restart_needed`` is ``True`` only for a cause on a PROVEN live
    receiver, ``None`` (unknown) for a cause on a pid whose liveness or
    identity could not be proven, else ``False`` — status must not print a
    verdict it did not earn. ``restart_reason`` is set whenever
    ``restart_needed`` is not ``False``.
    """

    record = _read(record_path(config_dir))
    if record is None:
        return None
    pid = record["pid"]
    alive = _receiver_alive(pid, config_dir)
    try:
        from local_operator.update import installed_version

        installed = installed_version()
    except Exception:  # noqa: BLE001 — a status field never fails the command
        installed = ""
    stale: bool | None = None
    if alive is not False:
        try:
            from local_operator.update import generation_staleness_of_process

            stale = generation_staleness_of_process(pid)
        except Exception:  # noqa: BLE001
            stale = None
    running_version = record.get("version") if isinstance(record.get("version"), str) else ""
    version_mismatch = bool(running_version and installed and running_version != installed)
    consent = record.get("consent_versions")
    reason: str | None = None
    if version_mismatch:
        reason = "version_mismatch"
    elif stale is True:
        reason = "replaced_build"
    if alive is False or reason is None:
        restart_needed: bool | None = False
        reason = None
    else:
        restart_needed = True if alive is True else None
    return {
        "pid": pid,
        "version": running_version,
        "generation": (
            record.get("generation") if isinstance(record.get("generation"), str) else None
        ),
        "consent_versions": (
            [v for v in consent if isinstance(v, str)] if isinstance(consent, list) else []
        ),
        "started_at": (
            record.get("started_at") if isinstance(record.get("started_at"), (int, float)) else None
        ),
        "singleton": record.get("singleton") if isinstance(record.get("singleton"), str) else None,
        "alive": alive,
        "installed_version": installed,
        "version_mismatch": version_mismatch,
        "generation_stale": stale,
        "restart_needed": restart_needed,
        "restart_reason": reason,
    }
