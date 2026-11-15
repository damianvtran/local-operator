"""The connector's own record of why it is not running.

A supervised connector cannot report anything about itself in-process: by the
time an operator looks, the process that knew is gone, and the file its
supervisor captured holds one line per restart with no timestamp and no context
(870 identical lines in the incident this module exists for). So the one durable
fact — *I stopped, deliberately, and this is why* — is a small JSON file beside
the connector's other state, and every surface reads that file instead of
forming its own opinion: `lop tunnel status`, the TUI, and the desktop route.

Only a PARK is written here. A transient failure keeps its existing behaviour
(exit 1, the supervisor's documented 10-second floor), deliberately: that line
is the operator's only evidence a live outage is still being retried, and
`state.json` is the vocabulary of "stopped and waiting for a person", which a
retry is not. The rate limit for the park line lives in this file rather than in
the process, because the process is exactly what a restart loop throws away.
"""

from __future__ import annotations

import datetime
import json
import time
from pathlib import Path
from typing import Any

# `gateway` is NOT imported here, deliberately: it is the connector's HTTP server
# and pulls jwt/starlette/httpx at module level, and this module is ALSO what the
# terminal's always-on five-second park poll imports (`tui/app.py`,
# `_poll_tunnel_park`). That poll runs on the event loop, so a module-level import
# here cost a cold first tick ~36 ms with nothing parked and ~340 ms with a park,
# stalling a conversation switch or a keystroke that landed on it (PR #3046
# review, m2). Only the WRITER, `mark_parked`, needs gateway's vocabulary, and it
# runs inside the connector, where gateway is already loaded; it imports it
# locally. The vocabulary stays in gateway rather than moving here because
# `TERMINAL_REMEDY` is keyed by gateway's reason codes and kept beside the
# sentences on purpose. `test_tunnel_park_notice.py` holds this to a fresh
# interpreter's `sys.modules`.
from local_operator.tunnels import config

#: A parked connector re-announces itself on one of these, never on every
#: attempt: a plist that is reloaded in a loop (or a reboot loop) must not be
#: able to fill the service log again, which is the failure this file repairs.
REPEAT_LOG_SECONDS = 900
REPEAT_LOG_ATTEMPTS = 50

#: The state word this file records. Named so a reader that finds a state it
#: does not know treats it as "not parked" rather than guessing.
PARKED = "parked"


def path() -> Path:
    return config.directory() / "state.json"


def read() -> dict[str, Any] | None:
    """This file's contents, or ``None`` for absent/unreadable/unusable.

    One answer for all of those on purpose: every caller wants "is there a park
    to act on?", a status command must never raise over a state file it happens
    to find, and a half-written file cannot exist (see `config.private_write`).
    """
    try:
        value = json.loads(path().read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def parked() -> dict[str, Any] | None:
    """The park record, when the connector is parked; else ``None``."""
    value = read()
    return value if value and value.get("state") == PARKED else None


def nag() -> dict[str, Any] | None:
    """The park a terminal should tell the user about, if there is one.

    Two gates, and both are about the user's intent rather than about this
    file: a machine with no tunnel enrolled has nothing to restore, and a
    tunnel the operator deliberately stopped (``config.json``'s ``stopped``) is
    one they are not using, so mentioning it is nagging about a decision they
    already made. A reader of this function can rely on the park NOT being
    actionable only through it — hence the narrow, named entry point rather
    than every caller re-deriving the rule.
    """
    record = parked()
    if record is None:
        return None
    try:
        value = config.load()
    except (OSError, ValueError):
        # No configuration on this device at all, or one nothing can parse:
        # there is no tunnel here for a user to sign back in to.
        return None
    return None if value.get("stopped") else record


def mark_parked(
    *,
    reason: str,
    detail: str,
    credential_id: int | None = None,
    now: int | None = None,
) -> bool:
    """Record that the connector stopped and why; True when this one is news.

    ``detail`` and the structured remedy are taken from `gateway`'s vocabulary
    rather than passed in, so the sentence this file carries and the sentence
    `lop tunnel status` prints are the same string by construction.
    """
    from local_operator.tunnels import gateway  # lazy: see the module imports

    at = int(time.time()) if now is None else now
    previous = parked()
    # Narrowed explicitly rather than through a `same` flag: the two facts below
    # (this cause's age, and whether the line is news) both depend on the
    # previous RECORD, not on an equality that has already been folded away.
    unchanged = previous is not None and previous.get("reason") == reason
    if unchanged and previous is not None:
        attempts = int(previous.get("attempts", 0)) + 1
        first_at = int(previous.get("first_at", at))
        logged_at = previous.get("logged_at")
        logged_attempts = previous.get("logged_attempts")
    else:
        attempts, first_at, logged_at, logged_attempts = 1, at, None, None
    announce = (
        not unchanged
        or not isinstance(logged_at, int)
        or not isinstance(logged_attempts, int)
        or at - logged_at >= REPEAT_LOG_SECONDS
        or attempts - logged_attempts >= REPEAT_LOG_ATTEMPTS
    )
    record: dict[str, Any] = {
        "state": PARKED,
        "reason": reason,
        "detail": detail,
        "remedy": {
            "command": gateway.TERMINAL_REMEDY.get(reason, "lop tunnel status"),
            "url": gateway.CONSOLE_URL,
        },
        "credential_id": credential_id,
        "at": at,
        "first_at": first_at,
        "attempts": attempts,
        "logged_at": at if announce else logged_at,
        "logged_attempts": attempts if announce else logged_attempts,
    }
    config.private_write(path(), json.dumps(record, indent=2) + "\n")
    return announce


def clear() -> None:
    """Withdraw the park: the connector is running, or the operator stopped it.

    Called on a successful connect and on an explicit local stop, so the file
    never outlives the condition it describes — a stale park would have every
    surface nag about a connector that is serving.
    """
    path().unlink(missing_ok=True)


#: How much of `service.log`'s TAIL is read for the connector's last exit line.
#: Bounded because the launchd log is only truncated by `lop tunnel install`, so
#: on a long-lived machine it can be large, and `status` must stay a cheap read.
#: A retry line is ~200 bytes and the connector writes at most one per restart
#: (the supervisor's 10 s floor), so 64 KiB holds the last ~300 lines — far more
#: than the one line this needs, and still ~0.1 ms to read.
LOG_TAIL_BYTES = 64 * 1024

#: The ONE line shape this reads: `service.main`'s retrying exit, which is this
#: package's own literal (`connector stopped reason=<code>…`), never an upstream
#: body. Parks are not read here — they live in `state.json`.
_SERVICE_PREFIX = " local_operator.tunnels.service: "
_RETRY_MARKER = _SERVICE_PREFIX + "connector stopped reason="

#: How old a retrying exit may be and still be the CURRENT cause.
#:
#: Derived from the retry cadence, not chosen for taste: a retrying connector
#: exits 1 and its supervisor restarts it no sooner than 10 s later
#: (`ThrottleInterval: 10` in `tunnels/install.py`'s launchd plan and
#: `restart_sec=10` in its systemd unit), and each failed start writes a fresh line. So while a
#: fault persists the newest line is at most ~10 s plus one start attempt old
#: (the control-plane calls are themselves bounded well under 30 s). 60 s is
#: several missed cycles of slack; past it nothing is restarting the connector,
#: and the line is history (review round 1, R1-1: a two-month-old line beside a
#: closed port had replaced the install advice).
CURRENT_FAILURE_S = 60


#: The four inactive-tunnel reason codes (`gateway.INACTIVE_REASONS`), spelled
#: here because this module must not import `gateway` (httpx, starlette) on the
#: TUI's poll path; `test_state_mirrors_the_gateway_inactive_reasons` holds the
#: two equal.
INACTIVE_REASONS = ("tunnel_inactive", "tunnel_disabled", "billing_inactive", "billing_lapsed")


def retrying_notice() -> dict[str, Any] | None:
    """A CURRENT retrying cause a person must act on, shaped like a park record.

    The TUI's card was raised from parks only (`nag`), but the four
    inactive-tunnel reasons are TRANSIENT by design (`service.classify_failure`:
    each clears from outside this process, so the connector keeps its 10 s
    retry floor and never parks). That left the card unreachable for exactly
    the causes #3945 names (round 1, D4/Q2). This reads the same current-exit
    evidence `report.probe` uses and returns ``{"reason", "remedy"}`` so the
    card's one renderer (`tui.app.tunnel_park_card`) can key it by kind; the
    remedy command is left to that renderer, which owns the in-app spelling.

    Only the inactive reasons qualify: the waits (`wait_network`,
    `wait_radient`) clear by themselves and a card would invite a "fix" for
    something that is not broken. The same intent gates as `nag`: no tunnel,
    or one the operator stopped, is never nagged about. A local file read,
    never the network.
    """
    if not (config.directory() / "config.json").exists():
        return None
    try:
        if config.load().get("stopped"):
            return None
    except (OSError, ValueError):
        return None
    last = last_connector_failure()
    if last is None or not last["current"] or last["reason"] not in INACTIVE_REASONS:
        return None
    return {"reason": last["reason"], "remedy": None}


def last_connector_failure(*, now: float | None = None) -> dict[str, Any] | None:
    """The connector's most recent retrying exit from `service.log`, if any.

    `state.json` deliberately records only PARKS (see `tunnels/state.py`), so a
    connector cycling on a transient fault — the field report's Radient 503 —
    leaves its cause only in the log its supervisor captures. This reads that
    cause back as a reason CODE (this package's vocabulary) plus the line's own
    timestamp; the free text after the code is not returned, so nothing beyond
    the fixed vocabulary reaches a status surface.

    ``current`` says whether the exit can explain what is happening NOW: it must
    be the LAST line the service wrote (a later line — a relay start, a park
    withdrawn — means the connector moved on) and no older than
    :data:`CURRENT_FAILURE_S`. `service.log` is truncated only by `lop tunnel
    install`, so without both checks one transient blip would explain every
    later outage forever.
    """
    log = config.directory() / "service.log"
    try:
        with log.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - LOG_TAIL_BYTES))
            tail = handle.read(LOG_TAIL_BYTES).decode("utf-8", "replace")
    except OSError:
        return None
    superseded = False
    for line in reversed(tail.splitlines()):
        stamp, marker, rest = line.partition(_RETRY_MARKER)
        if not marker:
            if _SERVICE_PREFIX in line:
                superseded = True
            continue
        code = rest.split(":", 1)[0].split(";", 1)[0].strip()
        if not code or not all(c.isalnum() or c == "_" for c in code):
            return None
        at: int | None
        try:
            at = int(datetime.datetime.fromisoformat(stamp.strip()).timestamp())
        except ValueError:
            at = None
        age = (time.time() if now is None else now) - at if at is not None else None
        # Both directions bounded: a stamp from the future (a clock moved back)
        # is no evidence of a retry happening now either.
        current = not superseded and age is not None and abs(age) <= CURRENT_FAILURE_S
        return {"reason": code, "at": at, "current": current}
    return None
