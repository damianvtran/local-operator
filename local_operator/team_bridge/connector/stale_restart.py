"""Restart a supervised receiver that is running code other than the installed ``lop``.

WHY THIS EXISTS (Team Bridge incident, 2026-11-17). A teammate's receiver was
jammed from 2026-11-13 17:39Z: launchd kept the process it had started on a
pre-v0.101.43 build, that build could not parse a threaded feed ref (fixed in
#3752), the all-or-nothing batch save refused with ``ref_migration_malformed``,
and the cursor never advanced. ``lop`` itself had long since been upgraded; the
fix was on disk, and the always-on process that needed it never re-imported it.
``receiver status`` (#4096) could say "Restart needed" — and restarted nothing.

WHY THE TWO EXISTING REPAIRS DID NOT FIRE. Both answer "is the process on the
generation ``current`` names?" (``launchd.restart_if_build_moved`` on the
upgrade path, ``connector.__main__._watch_own_generation`` in-process). On a
machine WITHOUT the generation layout — a ``uv tool`` or pipx install, which is
what the operator's own launcher is — there is no generation to compare, so
both decline by design, and the receiver keeps its build forever. The question
this module asks is the one that exists on every install: does the build the
RECEIVER recorded (``runtime_record``) match the build ``lop`` would run now?

THE DECISION, and why each arm is shaped the way it is (:func:`decide`):

* **Only what ``receiver status`` already calls "restart needed", and only a
  PROVEN one.** The verdict is ``connector.cli._restart_verdict`` — the same
  function status prints — so the product cannot restart on a reading it would
  not show. ``version_mismatch`` acts. ``no_record`` acts too, and is the arm
  that would have caught the incident: ``runtime_record`` shipped in v0.106.6,
  so every receiver started by older code (the jammed one included) writes NO
  record, and a supervised pid with no record is exactly that receiver.
  ``replaced_build`` is left to ``restart_if_build_moved``, which owns it.
* **The supervisor's pid, and the record's pid, must be the same process.**
  A restart goes through the service manager, so it can only replace the
  process the service manager is running. A record naming some other pid (a
  foreground receiver someone started in a terminal) is not this repair's to
  touch: a kick would bounce the healthy supervised one and leave the stale one.
* **Uptime grace (:data:`MIN_UPTIME_S`).** A receiver writes its record at
  start (``runtime_record.record_start``), so a young process with no record
  may simply not have written it yet, and the upgrade path's own kick produces
  exactly such a process a moment before this asks. Five minutes is the
  self-check's flap guard (``SELF_CHECK_MIN_UPTIME_S``), reused rather than
  re-derived: the same "well past any start or flip" bound for the same daemon.
  An age that cannot be read is treated as young — no restart.
* **Idempotent and rate-limited** through one disposable side file
  (:func:`marker_path`, the ``runtime_record`` pattern): never a second kick
  for the same (pid, installed version) — if that kick did not take, asking
  again is a loop, not a repair — and never more than one kick per
  :data:`MIN_INTERVAL_S` whatever changed. Callers hold the service-sync lock
  (``install._sync_exclusive``), so two surfaces starting together read and
  write the marker one at a time.

HOW IT RESTARTS. Through the receiver's OWN service manager and the guarded
helpers ``install`` already uses — ``launchd.kickstart`` (``kickstart -k`` on
the label; correct here because the plist is current, see
``launchd.restart_if_build_moved``) or ``systemctl --user restart`` on the
unit. Never a signal by pid, never a kill by program name. Both helpers refuse
a redirected ``HOME`` (``launchd.is_own_plist``,
``supervisors.systemd_unit_is_addressable``), so a sandboxed run cannot bounce
the operator's receiver. Windows declines: Task Scheduler has no restart verb
this module may use without a stop/start pair, and that is ``install``'s
territory (``_stop_task``), not a catch-up's.

ONE LOG LINE, NO CONTENT: the closed outcome code, the trigger and the closed
reason — never a version string, path, pid or account. Never raises.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: The uptime grace. See the module docstring; the value is the connector's own
#: self-check flap guard (``connector.__main__.SELF_CHECK_MIN_UPTIME_S``).
MIN_UPTIME_S = 300.0

#: The floor between two kicks, whatever changed. Surface starts and upgrades
#: are the only callers, so this bounds a pathological case (a shim that keeps
#: resolving an old build, a crash right after start) to two restarts an hour;
#: it is not on any healthy path's critical path.
MIN_INTERVAL_S = 1800.0

#: The verdicts this repair acts on (``connector.cli._restart_verdict``).
ACTIONABLE_REASONS = frozenset({"version_mismatch", "no_record"})

# Closed outcome codes (the log line and the tests read these).
RESTARTED = "restarted"
NOT_NEEDED = "not_needed"
UNPROVEN = "unproven"
OTHER_PROCESS = "other_process"
OTHER_REPAIR = "other_repair"
TOO_YOUNG = "too_young"
ALREADY_KICKED = "already_kicked"
RATE_LIMITED = "rate_limited"
UNSUPPORTED = "unsupported"
REFUSED = "refused"
FAILED = "failed"


@dataclass(frozen=True)
class Observation:
    """Everything :func:`decide` needs, read once so the decision stays pure."""

    restart_needed: bool | None
    reason: str | None
    #: The pid the service manager runs, or ``None`` when it could not be read.
    supervised_pid: int | None
    #: The pid the runtime record names, or ``None`` when there is no record.
    recorded_pid: int | None
    #: Seconds the supervised process has been up, or ``None`` when unreadable.
    uptime_s: float | None
    installed_version: str
    #: The marker (``{"at", "pid", "installed", "reason", "probed"}``) or ``None``.
    last_kick: dict[str, Any] | None


def decide(observation: Observation, *, now: float) -> str:
    """The closed answer: :data:`RESTARTED` means "kick it"; anything else declines."""

    o = observation
    if o.restart_needed is False or o.reason is None:
        return NOT_NEEDED
    if o.reason not in ACTIONABLE_REASONS:
        # ``replaced_build``: ``launchd.restart_if_build_moved`` and the
        # in-process self-check own it; a second actor would double-kick.
        return OTHER_REPAIR
    if o.restart_needed is not True or o.supervised_pid is None:
        return UNPROVEN
    if o.reason == "version_mismatch" and o.recorded_pid != o.supervised_pid:
        return OTHER_PROCESS
    if o.uptime_s is None or o.uptime_s < MIN_UPTIME_S:
        return TOO_YOUNG
    last = o.last_kick
    if last is not None and "at" in last:
        if last.get("pid") == o.supervised_pid and last.get("installed") == o.installed_version:
            return ALREADY_KICKED
        at = last.get("at")
        if isinstance(at, (int, float)) and 0 <= now - at < MIN_INTERVAL_S:
            return RATE_LIMITED
    return RESTARTED


def marker_path(config_dir: Path | str) -> Path:
    """The last-kick marker, beside the runtime record (disposable side file)."""

    return Path(config_dir) / "team_bridge" / "connector-restart.json"


def _read_marker(config_dir: Path | str) -> dict[str, Any] | None:
    try:
        payload = json.loads(marker_path(config_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_marker(config_dir: Path | str, payload: dict[str, Any]) -> None:
    path = marker_path(config_dir)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    except OSError:
        # A marker that could not be written costs at most one extra kick on
        # the next surface start; it must never fail the caller.
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _supervised_pid() -> int | None:
    """The pid the service manager runs for the receiver, through guarded reads only."""

    from local_operator import launchd, supervisors
    from local_operator.team_bridge.connector import install

    kind = supervisors.supervisor()
    if kind == supervisors.LAUNCHCTL:
        return launchd.job_pid(
            label=install.LABEL, path=install.plist_path(), run=install._launchctl
        )
    if kind == supervisors.SYSTEMCTL:
        if not supervisors.systemd_unit_is_addressable(install.SYSTEMD_UNIT):
            return None
        result = install._systemctl("show", "-p", "MainPID", "--value", install.SYSTEMD_UNIT)
        try:
            pid = int((result.stdout or "").strip()) if result.returncode == 0 else 0
        except ValueError:
            return None
        return pid if pid > 0 else None
    return None


def _uptime_s(pid: int, running: dict[str, Any] | None, *, now: float) -> float | None:
    """How long ``pid`` has been up: its own recorded start, else ``ps``'s ``lstart``."""

    if running is not None and running.get("pid") == pid:
        started = running.get("started_at")
        if isinstance(started, (int, float)):
            return now - started
    from local_operator.team_bridge.connector import reap

    process = reap._read_process(pid)
    if process is None:
        return None
    try:
        # ``reap._run`` pins LC_ALL=C, so lstart is always this English shape.
        started = time.mktime(time.strptime(process.start, "%a %b %d %H:%M:%S %Y"))
    except (ValueError, OverflowError):
        return None
    return now - started


def observe(config_dir: Path | str, *, now: float) -> Observation:
    """Read the verdict, the supervisor's pid, the uptime and the marker."""

    from local_operator.team_bridge.connector import cli, install

    running = cli._running_receiver(Path(config_dir))
    held, held_pid = cli._supervised_receiver(install.connector_state(Path(config_dir)))
    restart_needed, reason = cli._restart_verdict(running, held=held, held_pid=held_pid)
    supervised = held_pid if held_pid is not None else (_supervised_pid() if held else None)
    if running is not None and running.get("installed_version"):
        installed = str(running["installed_version"])
    else:
        try:
            from local_operator.update import installed_version

            installed = installed_version()
        except Exception:  # noqa: BLE001 — an unreadable version only weakens idempotency
            installed = ""
    return Observation(
        restart_needed=restart_needed,
        reason=reason,
        supervised_pid=supervised,
        recorded_pid=running.get("pid") if running is not None else None,
        uptime_s=_uptime_s(supervised, running, now=now) if supervised else None,
        installed_version=installed,
        last_kick=_read_marker(config_dir),
    )


def _kick() -> str:
    """Ask the receiver's own service manager to restart it; a closed code."""

    from local_operator import launchd, supervisors
    from local_operator.team_bridge.connector import install

    kind = supervisors.supervisor()
    if kind == supervisors.LAUNCHCTL:
        ok = launchd.kickstart(
            label=install.LABEL, path=install.plist_path(), run=install._launchctl
        )
        return RESTARTED if ok else REFUSED
    if kind == supervisors.SYSTEMCTL:
        if not supervisors.systemd_unit_is_addressable(install.SYSTEMD_UNIT):
            return REFUSED
        result = install._systemctl("restart", install.SYSTEMD_UNIT)
        return RESTARTED if result.returncode == 0 else REFUSED
    return UNSUPPORTED


def _installed() -> str:
    try:
        from local_operator.update import installed_version

        return installed_version()
    except Exception:  # noqa: BLE001 — unreadable means "cannot compare"
        return ""


def note_fresh_start(config_dir: Path | str) -> None:
    """The supervisor just started the receiver on the installed build. Never raises.

    Called after an ``installed``/``registered`` sync: the process it started
    runs THIS build, so the next surface start need not probe for a receiver
    older than the runtime record (:func:`_cheap_skip`).
    """

    try:
        installed = _installed()
        if installed:
            marker = _read_marker(config_dir) or {}
            _write_marker(config_dir, {**marker, "probed": installed})
    except Exception:  # noqa: BLE001
        pass


def _cheap_skip(config_dir: Path | str, installed: str) -> bool:
    """The healthy path: answer from two small files, with no supervisor or ``ps`` call.

    WHY. ``sync_connector_service`` runs at every surface start and its no-op
    cost is a contract ("one file read and one supervisor query", pinned by
    ``test_a_second_reconciliation_is_a_noop`` and the per-platform repeat
    tests). So the full probe runs only when there is something to ask about:

    * a record naming the installed version is a current receiver — skip;
    * no record at all is the "older than the record" case, but a machine where
      the receiver is not running has no record either, so that probe runs ONCE
      per installed version (``probed`` in the marker) unless it could not
      reach a final answer (a too-young receiver is asked again next start);
    * a record naming another version always earns the probe — that is the
      jam this module exists for, and :func:`decide` rate-limits the kick.
    """

    if not installed:
        return True
    record = _runtime_record(config_dir)
    if record is not None:
        return record.get("version") == installed
    marker = _read_marker(config_dir) or {}
    return marker.get("probed") == installed


def _runtime_record(config_dir: Path | str) -> dict[str, Any] | None:
    from local_operator.team_bridge.connector import runtime_record

    return runtime_record._read(runtime_record.record_path(config_dir))


#: Decisions that settle the "receiver older than the record" question for this
#: installed version. ``too_young``/``unproven`` are not final: the receiver may
#: be mid-start, or its pid unreadable right now.
_FINAL = frozenset({RESTARTED, NOT_NEEDED, OTHER_PROCESS, OTHER_REPAIR, ALREADY_KICKED})


def restart_if_stale(config_dir: Path | str, *, trigger: str) -> str:
    """Restart the supervised receiver when it provably runs other code. Never raises.

    ``trigger`` names the caller for the log line (``surface`` today). The
    caller holds ``install._sync_exclusive`` (see the module docstring).
    """

    now = time.time()
    try:
        installed = _installed()
        if _cheap_skip(config_dir, installed):
            return NOT_NEEDED
        observation = observe(config_dir, now=now)
        code = decide(observation, now=now)
        reason = observation.reason or "none"
        marker = dict(observation.last_kick or {})
        if code == RESTARTED:
            code = _kick()
            if code == RESTARTED:
                marker.update(
                    {
                        "at": now,
                        "pid": observation.supervised_pid,
                        "installed": observation.installed_version,
                        "reason": reason,
                    }
                )
        if code in _FINAL and installed and marker.get("probed") != installed:
            marker["probed"] = installed
        if marker != (observation.last_kick or {}):
            _write_marker(config_dir, marker)
    except Exception as exc:  # noqa: BLE001 — a catch-up never fails its caller
        logger.warning(
            "bridge connector stale restart: trigger=%s code=%s error=%s",
            trigger,
            FAILED,
            type(exc).__name__,
        )
        return FAILED
    if code != NOT_NEEDED:
        # ONE line, closed vocabulary only. INFO for the acted/declined-with-
        # reason cases; nothing at all on the healthy path.
        level = logging.WARNING if code == REFUSED else logging.INFO
        logger.log(
            level,
            "bridge connector stale restart: trigger=%s code=%s reason=%s",
            trigger,
            code,
            reason,
        )
    return code
