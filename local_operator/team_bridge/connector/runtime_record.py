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
  process is now the one serving this config dir;
* on a clean exit, remove the record only when it still names our pid.

A SIGKILL'd receiver leaves its record behind; the reader checks liveness (and,
where ``ps`` answers, that the pid is still a receiver, so a reused pid is not
mistaken for one) and reports "has exited" rather than a running build.

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
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    except OSError as error:
        logger.warning("bridge connector runtime record not written: %s", type(error).__name__)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def record_start(config_dir: Path | str, *, started_at: float) -> None:
    """At process start: record ourselves unless a live receiver already is. Never raises.

    Forks ``ps`` (generation and peer liveness) once per process start, before
    any network call — the same one-shot cost the start path already pays.
    """

    path = record_path(config_dir)
    try:
        existing = _read(path)
        if existing is not None and existing["pid"] != os.getpid():
            if _receiver_alive(existing["pid"]) is not False:
                return
        _write(path, _own_facts(started_at=started_at, singleton=SINGLETON_WAITING))
    except Exception as error:  # noqa: BLE001
        logger.warning("bridge connector runtime record skipped: %s", type(error).__name__)


def record_singleton_held(config_dir: Path | str, *, started_at: float) -> None:
    """After the singleton is acquired: this process now serves the config dir."""

    try:
        _write(record_path(config_dir), _own_facts(started_at=started_at, singleton=SINGLETON_HELD))
    except Exception as error:  # noqa: BLE001
        logger.warning("bridge connector runtime record skipped: %s", type(error).__name__)


def clear_own_record(config_dir: Path | str) -> None:
    """On a clean exit: remove the record only if it still names this pid."""

    path = record_path(config_dir)
    try:
        existing = _read(path)
        if existing is not None and existing["pid"] == os.getpid():
            path.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass


def _receiver_alive(pid: int) -> bool | None:
    """``True`` live receiver, ``False`` gone (or a reused pid), ``None`` unsure.

    ``ps`` confirms the pid is still running the receiver module, so a pid the
    kernel handed to an unrelated process after a SIGKILL is not reported as a
    running build. Where ``ps`` cannot answer (Windows, a sandbox) liveness
    alone decides, failing toward "alive" like :func:`procstate.pid_alive`.
    """

    from local_operator.procstate import pid_liveness

    if pid <= 0:
        return False
    alive = pid_liveness(pid)
    if alive is False:
        return False
    from local_operator.team_bridge.connector import reap

    if not reap._is_posix():
        return alive
    process = reap._read_process(pid)
    if process is None:
        return alive
    return reap._is_connector(process.argv)


def running_receiver(config_dir: Path | str) -> dict[str, Any] | None:
    """The recorded receiver, annotated for ``status``; ``None`` when none recorded.

    Adds ``alive`` (``True``/``False``/``None`` = could not tell),
    ``installed_version`` (this reading process's — i.e. the installed ``lop``'s
    — version), ``generation_stale`` (``update.generation_staleness_of_process``,
    tri-state, only for a live pid) and ``restart_needed``: ``True`` only when
    the receiver is live AND either its version differs from a known installed
    version or its generation is provably not ``current``. Unknown stays
    ``False`` — status must not print a verdict it did not earn.
    """

    record = _read(record_path(config_dir))
    if record is None:
        return None
    pid = record["pid"]
    alive = _receiver_alive(pid)
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
        "restart_needed": alive is not False and (version_mismatch or stale is True),
    }
