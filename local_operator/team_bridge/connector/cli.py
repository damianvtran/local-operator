"""``lop team-bridge receiver status|install|remove|logs`` — the backend-only door.

WHY THIS EXISTS. The receiver (the always-on connector process,
``python -m local_operator.team_bridge.connector``) is placed automatically at
every surface's start and at every enrolment/enable/Turn-off event
(:mod:`local_operator.team_bridge.connector.install`). A machine with no
desktop app — a server, a headless Linux box, a Windows machine driven over
SSH — still needs a way to see whether the receiver is running, place it on
demand, take it down and read its log. These four verbs are that, for a person
or for an agent following ``guide://team-bridge``.

THE ONE RULE: no second installer. ``install`` IS
:func:`~local_operator.team_bridge.connector.install.sync_connector_service`
— the same guards (a supervisor must exist; only the real user's default
config root; never from a redirected home; never from a source checkout), the
same wanted-state decision inside the same lock. So ``install`` never enrols
and never turns anything on: on a machine that is not enrolled, or has agent
messages turned off, it changes nothing and says what to do. ``remove`` is
:func:`~local_operator.team_bridge.connector.install.remove_connector_service`
— the same locked body with the not-wanted answer forced, leaving the wanted
STATE alone (and saying so, because the next start would place it again).

REGISTRATION lives in :mod:`local_operator.team_bridge_arguments` (stdlib-only,
run on every ``lop`` start); this module is imported only once a
``team-bridge`` verb is dispatched, and imports the installer and the store
lazily even then.

COPY. Every line a person reads is keyed (``cli.team_bridge`` namespace);
paths, commands and config keys are interpolated verbatim. ``--json`` emits
stable machine keys (``code``, ``state``, ...) for agents and scripts.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from local_operator.i18n.keys import cli_team_bridge as _k
from local_operator.team_bridge_arguments import DEFAULT_LOG_LINES


def main(args: argparse.Namespace) -> int:
    """Dispatch a ``lop team-bridge`` invocation; exit code 0 on a settled answer."""

    if getattr(args, "team_bridge_command", None) != "receiver":
        print("usage: lop team-bridge receiver {status,install,remove,logs}")
        return 2
    verb = getattr(args, "receiver_command", None)
    from local_operator.paths import config_dir

    root = config_dir()
    if verb == "status":
        return _status(root, as_json=bool(getattr(args, "json", False)))
    if verb in ("install", "remove"):
        return _change(root, verb, as_json=bool(getattr(args, "json", False)))
    if verb == "logs":
        return _logs(root, max(1, int(getattr(args, "lines", DEFAULT_LOG_LINES))))
    print("usage: lop team-bridge receiver {status,install,remove,logs}")
    return 2


def _tr() -> Any:
    from local_operator.i18n.text import tr

    return tr("cli.team_bridge")


# -- status --------------------------------------------------------------------


def _enrolment_state(root: Path) -> str:
    """``on`` / ``off`` / ``not_enrolled`` / ``unknown`` — read-only, never creates.

    The same two facts ``install._wanted`` decides on, named for a person.
    """

    from local_operator.team_bridge.connector import install

    wanted = install._wanted(root)
    if wanted is None:
        return "unknown"
    if wanted:
        return "on"
    from local_operator.team_bridge.store import Store

    try:
        owner = Store(root).active_owner_readonly()
    except Exception:  # noqa: BLE001 — a status line never fails the command
        return "unknown"
    return "not_enrolled" if owner is None else "off"


def _last_check(root: Path) -> float | None:
    """When the receiver last merged an inbox page (``sync.last_observed_at``).

    Read-only over an EXISTING store (``mode=ro``): a never-enrolled machine
    has no database and this must not create one. ``None`` when unrecorded.
    The value moves on an inbox merge or a completed resync — the walk the
    receiver runs at start and after a gap — so it answers "has this receiver
    talked to the hub", not "did it poll a second ago" (a quiet long poll that
    returns nothing writes no row, by design: no write per poll).
    """

    import sqlite3

    from local_operator.team_bridge.store import state_db_path

    path = state_db_path(root)
    if not path.exists():
        return None
    try:
        conn = sqlite3.connect(f"{path.absolute().as_uri()}?mode=ro", uri=True, timeout=1.0)
    except sqlite3.Error:
        return None
    try:
        row = conn.execute("SELECT MAX(last_observed_at) FROM sync").fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    value = row[0] if row else None
    return float(value) if isinstance(value, (int, float)) else None


def _compact_age(seconds: float) -> str:
    """``41m`` / ``2h 10m`` / ``30s``: minute precision once past a minute."""

    from local_operator.i18n.format import format_duration
    from local_operator.i18n.text import active_locale

    whole = int(seconds)
    return format_duration(whole if whole < 60 else whole - whole % 60, active_locale())


def _counts(mapping: dict[str, int]) -> str:
    # Codes and categories are technical identifiers (closed vocabularies),
    # interpolated verbatim; only the sentence around them is localized.
    return ", ".join(f"{name} {count}" for name, count in mapping.items())


def _screening_line(screening: dict[str, Any], *, now: float) -> Any:
    """The ONE human screening line as a keyed message (rendered by the caller)."""

    held = screening["held"]
    breaker = screening["breaker"]
    rejected = screening["rejected_24h"]
    if held["count"] == 0 and rejected["count"] == 0 and breaker["failures"] == 0:
        return _k.status_screening_healthy()
    t = _tr()
    if held["count"] == 0:
        held_text = t(_k.status_screening_held_none())
    else:
        age = _compact_age(held["oldest_age_s"] or 0.0)
        after = held.get("after_screening") or {}
        # Screened-but-undelivered holds ride the SAME verbatim code list as
        # their own ``held_after_screening <n>`` entry (a subset of the codes
        # before it), so the bucket is visible with no new localized copy.
        codes = _counts(
            {
                **held["by_code"],
                **({"held_after_screening": after["count"]} if after.get("count") else {}),
            }
        )
        failure_class = held.get("dominant_failure_class")
        held_text = t(
            _k.status_screening_held_class(
                count=held["count"], codes=codes, age=age, failure_class=failure_class
            )
            if failure_class
            else _k.status_screening_held(count=held["count"], codes=codes, age=age)
        )
    if breaker["state"] == "open":
        state = "open"
    else:
        state = "failing" if breaker["failures"] else "closed"
    remaining = _compact_age(max(0.0, (breaker["open_until"] or now) - now))
    breaker_text = t(
        _k.status_screening_breaker(state=state, remaining=remaining, failures=breaker["failures"])
    )
    rejected_text = (
        t(
            _k.status_screening_rejected(
                count=rejected["count"], categories=_counts(rejected["by_category"])
            )
        )
        if rejected["count"]
        else t(_k.status_screening_rejected_none())
    )
    return _k.status_screening(held=held_text, breaker=breaker_text, rejected=rejected_text)


def _status(root: Path, *, as_json: bool) -> int:
    import os

    from local_operator import supervisors
    from local_operator.team_bridge.connector import install
    from local_operator.team_bridge.store import read_screening_status

    state = install.connector_state(root)
    enrolment = _enrolment_state(root)
    last = _last_check(root)
    screening = read_screening_status(root)
    log = install.log_path(root)
    journal = install.journal_hint()
    linger: bool | None = None
    if supervisors.supervisor() == supervisors.SYSTEMCTL:
        linger = install._linger_enabled()
    user = os.environ.get("USER") or "$USER"
    linger_command = f"loginctl enable-linger {user}"
    if as_json:
        print(
            json.dumps(
                {
                    "state": state,
                    "supervisor": supervisors.supervisor() or "none",
                    "registration": install.supervisor_name(),
                    "enrolment": enrolment,
                    "config_dir": str(root),
                    "log": str(log),
                    "journal": journal,
                    "last_check": last,
                    "linger": linger,
                    "foreground_command": install.foreground_command(root),
                    # ``None`` when not enrolled or unreadable; see _screening.
                    "screening": screening,
                }
            )
        )
        return 0
    t = _tr()
    print(t(_k.status_supervisor(name=install.supervisor_name())))
    print(t(_k.status_state(state=state)))
    print(t(_k.status_enrolment(state=enrolment)))
    print(t(_k.status_config_dir(path=str(root))))
    print(t(_k.status_log(path=str(log))))
    if journal:
        print(t(_k.status_journal(command=journal)))
    if last is None:
        print(t(_k.status_last_check_never()))
    else:
        when = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(last))
        print(t(_k.status_last_check(when=when)))
    if screening is not None:
        print(t(_screening_line(screening, now=time.time())))
    if linger is not None:
        print(t(_k.status_linger(state="yes" if linger else "no", command=linger_command)))
    if state == install.CODE_ISOLATED:
        print(t(_k.status_isolated_note(command=install.foreground_command(root))))
    elif state == install.CODE_NO_SUPERVISOR:
        print(t(_k.status_no_supervisor_note(command=install.foreground_command(root))))
    return 0


# -- install / remove -----------------------------------------------------------


def _change(root: Path, verb: str, *, as_json: bool) -> int:
    from local_operator.team_bridge.connector import install

    outcome = (
        install.sync_connector_service(root)
        if verb == "install"
        else install.remove_connector_service(root)
    )
    # ``install`` on a machine that does not want a receiver answers ``absent``
    # (nothing installed) or ``removed`` (a stale unit taken down): either way
    # the person asked for an install and got none, so they are told WHY and
    # what to do — the command never enrols or turns anything on itself.
    not_wanted = verb == "install" and outcome.code in (install.CODE_ABSENT, install.CODE_REMOVED)
    ok = (
        outcome.code
        in (
            install.CODE_INSTALLED,
            install.CODE_CURRENT,
            install.CODE_REGISTERED,
            install.CODE_REMOVED,
            install.CODE_ABSENT,
        )
        and not not_wanted
    )
    if as_json:
        print(
            json.dumps(
                {
                    "ok": ok,
                    "code": outcome.code,
                    "detail": outcome.detail,
                    "not_wanted": not_wanted,
                    "state": install.connector_state(root),
                }
            )
        )
        return 0 if ok else 1
    t = _tr()
    for line in _outcome_lines(outcome, root=root, verb=verb, not_wanted=not_wanted):
        print(t(line))
    return 0 if ok else 1


def _outcome_lines(outcome: Any, *, root: Path, verb: str, not_wanted: bool) -> list[Any]:
    """The keyed sentences for one outcome; the detail rides verbatim."""

    from local_operator import launchd
    from local_operator.team_bridge.connector import install

    code, detail = outcome.code, outcome.detail or ""
    if not_wanted:
        return [_k.outcome_not_wanted(), _k.outcome_not_wanted_hint()]
    simple = {
        install.CODE_INSTALLED: _k.outcome_installed,
        install.CODE_CURRENT: _k.outcome_current,
        install.CODE_REGISTERED: _k.outcome_registered,
        install.CODE_ABSENT: _k.outcome_absent,
        install.CODE_SOURCE_CHECKOUT: _k.outcome_source_checkout,
        install.CODE_NOT_ADDRESSABLE: _k.outcome_not_addressable,
        install.CODE_UNKNOWN: _k.outcome_unknown,
    }
    lines: list[Any] = []
    if code == install.CODE_REMOVED:
        lines.append(_k.outcome_removed())
        if verb == "remove":
            lines.append(_k.outcome_remove_hint())
    elif code in simple:
        lines.append(simple[code]())
    elif code == install.CODE_NO_SUPERVISOR:
        lines.append(_k.outcome_no_supervisor(detail=detail))
    elif code == install.CODE_NO_USER_MANAGER:
        lines.append(_k.outcome_no_user_manager(detail=detail))
    elif code == install.CODE_ISOLATED:
        # The guard's own sentence is shared with the wake supervisor and talks
        # about wakes; the person here needs the two folders and the way to run
        # the receiver for theirs.
        default = launchd.default_config_root()
        lines.append(
            _k.outcome_isolated(
                path=str(root),
                default=str(default) if default is not None else "?",
                command=install.foreground_command(root),
            )
        )
    else:
        lines.append(_k.outcome_failed(detail=detail or code))
    if code in (install.CODE_INSTALLED, install.CODE_REGISTERED) and detail:
        # The Linux arm's lingering note (read back from ``loginctl``, never
        # assumed): an English technical sentence, carried verbatim as the
        # other installer details are.
        lines.append(_k.outcome_note(detail=detail))
    return lines


# -- logs ------------------------------------------------------------------------


def _logs(root: Path, lines: int) -> int:
    from local_operator.team_bridge.connector import install

    t = _tr()
    shown = False
    for path in (install.log_path(root), install.stderr_path(root)):
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        tail = text.splitlines()[-lines:]
        if not tail:
            continue
        print(t(_k.logs_header(path=str(path))))
        for line in tail:
            print(line)
        shown = True
    if not shown:
        print(t(_k.logs_none(path=str(install.log_path(root)))))
    journal = install.journal_hint()
    if journal:
        print(t(_k.status_journal(command=journal)))
    return 0
