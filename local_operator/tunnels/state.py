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

import json
import time
from pathlib import Path
from typing import Any

from local_operator.tunnels import config, gateway

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
    at = int(time.time()) if now is None else now
    previous = parked()
    same = bool(previous) and previous.get("reason") == reason
    attempts = int(previous.get("attempts", 0)) + 1 if same else 1
    first_at = int(previous["first_at"]) if same and "first_at" in previous else at
    logged_at = previous.get("logged_at") if same else None
    logged_attempts = previous.get("logged_attempts") if same else None
    announce = (
        not same
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
