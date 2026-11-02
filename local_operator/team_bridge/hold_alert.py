"""The long-hold alert: tell the owner, once, that held messages have waited too long.

WHY. A conversation message that is held (in screening, or admitted and waiting
for delivery) is silent to everyone it concerns: the agent reads ``body: null``
and the owner sees nothing. When screening is down the holds pile up for hours.
``tick`` — called from the connector loop — counts the conversation holds older
than ``team_bridge.hold_alert_after_minutes`` and keeps ONE keyed notification
in the designated session's inbox in step with that count.

WHY IT RUNS IN ``run_forever`` AND NOT IN THE ITERATION. ``_run_iteration_inner``
is skipped while the hub is unreachable, which is exactly when holds age.

CONTENT LAW. The notification carries a count and the owner's own threshold:
no message id, sender, conversation, body, hold code or age. The age is left
out on purpose so the notification's content hash moves only when the count
does; a growing age would write a new revision (and re-notify) on every tick.

CLEARING. One keyed update at priority ``low``, sent to the session that RAISED
the alert (recorded per process, so a changed designated session cannot strand
it) and only while that keyed item still exists: a pruned row is not re-created
as a stray "cleared" item. A fresh process has no record, so it looks the keyed
item up in the designated session's inbox and clears a standing ``normal`` one;
that is how a restart (or a threshold of ``0``) cannot leave a stale alert up.
The notification store treats a later ``low`` -> ``normal`` as an escalation that
revives a dismissed row, so a new incident reaches the owner again even if they
dismissed the last one.

Every outcome is a closed code, and nothing here raises: a notification is
chrome, and a bug here must never stop the receiver.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

#: Minimum gap between runs per endpoint. The loop spins every few seconds; the
#: alert needs minutes-level resolution, and each run is a few SQLite reads.
THROTTLE_S = 300.0

#: ``endpoint_id`` -> monotonic stamp of the last run that got past the throttle.
_last_tick: dict[str, float] = {}

#: ``endpoint_id`` -> the session whose inbox THIS process raised the alert in, until
#: it clears it. The session is kept because the designated session can change
#: while an alert stands, and the clear must reach the inbox that shows it.
_raised: dict[str, str] = {}


def _key(endpoint_id: str) -> str:
    return f"team-bridge-long-hold:{endpoint_id}"


def _threshold_minutes(config_dir: Path) -> int | None:
    """The threshold read fresh from disk, or ``None`` when the file cannot be read now.

    Read per run, never cached: the owner can change or disable the alert and it
    must follow without a restart. The same ``read_config_values`` the wake
    fence uses, so a torn read is ``None`` (skip this run), not "unset".
    """

    from local_operator.config import read_config_values
    from local_operator.team_bridge.settings import TeamBridgeSettings

    values = read_config_values(config_dir)
    if values is None:
        return None
    return TeamBridgeSettings.from_values(values).hold_alert_after_minutes


async def tick(
    owner: Any,
    config_dir: Path | str,
    *,
    now: float | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> str:
    """Bring the long-hold notification in line with the store; return a closed outcome.

    ``throttled`` | ``disabled`` | ``no_session`` | ``none_held`` | ``raised`` |
    ``updated`` | ``unchanged`` | ``cleared`` | ``failed``. ``now`` (wall clock)
    and ``monotonic`` are injectable for tests.
    """

    endpoint_id = str(getattr(owner, "endpoint_id", ""))
    stamp = monotonic()
    last = _last_tick.get(endpoint_id)
    if last is not None and 0 <= stamp - last < THROTTLE_S:
        return "throttled"
    _last_tick[endpoint_id] = stamp
    try:
        return await asyncio.to_thread(_run, owner, Path(config_dir), now)
    except Exception:  # noqa: BLE001 — a notification is chrome; the receiver goes on
        logger.info("bridge: long-hold alert skipped code=failed")
        return "failed"


def _run(owner: Any, config_dir: Path, now: float | None) -> str:
    """The synchronous body of :func:`tick` (store and notification I/O)."""

    from local_operator.i18n.keys import cli_team_bridge_alert as keys
    from local_operator.i18n.text import tr
    from local_operator.notifications.store import NotificationStore
    from local_operator.team_bridge.settings import DEFAULT_HOLD_ALERT_AFTER_MINUTES
    from local_operator.team_bridge.store import Store

    minutes = _threshold_minutes(config_dir)
    if minutes is None:  # a torn config read: skip this run, do not guess
        return "failed"
    endpoint_id = str(owner.endpoint_id)
    recorded = _raised.get(endpoint_id)
    store = Store(config_dir)
    session_id = store.designated_session_for_readonly(owner)
    if not session_id and recorded is None:
        # Nothing to raise into and nothing this process raised. (A restart with
        # no designated session has no inbox to look in either.)
        return "disabled" if minutes == 0 else "no_session"
    notifications = NotificationStore(config_dir)
    t = tr("cli.team_bridge_alert")

    def clear(target: str, shown_minutes: int) -> bool:
        """Send the one low update, only to an item that is still standing."""

        item = notifications.get_by_key(target, _key(endpoint_id))
        if item is None or item["priority"] != "normal":
            return False
        notifications.notify(
            target,
            title=t(keys.cleared_title()),
            body_md=t(keys.cleared_body(minutes=shown_minutes)),
            priority="low",
            key=_key(endpoint_id),
        )
        return True

    # Where an alert could be standing: the inbox this process raised it in, and
    # the designated session (the only place a previous process could have put it).
    inboxes = [target for target in dict.fromkeys((recorded, session_id)) if target]

    if minutes == 0:
        # Disabled: a standing alert would otherwise outlive the setting. The
        # cleared copy needs a threshold, so it names the default one.
        for target in inboxes:
            clear(target, DEFAULT_HOLD_ALERT_AFTER_MINUTES)
        _raised.pop(endpoint_id, None)
        return "disabled"

    found = store.long_holds_readonly(owner, older_than_s=minutes * 60.0, now=now)
    count = int(found["count"])

    if count > 0:
        if not session_id:
            return "no_session"
        if recorded is not None and recorded != session_id:
            clear(recorded, minutes)  # the old inbox must not keep a stale alert
            _raised.pop(endpoint_id, None)
        outcome = notifications.notify(
            session_id,
            title=t(keys.long_hold_title()),
            body_md=t(keys.long_hold_body(count=count, minutes=minutes)),
            priority="normal",
            key=_key(endpoint_id),
        )
        # ``_raised`` also records an alert this process found already in place
        # (``unchanged`` after a restart): it asserted it, so it clears it.
        newly = endpoint_id not in _raised
        _raised[endpoint_id] = session_id
        if outcome.action == "unchanged":
            return "unchanged"
        if outcome.action in ("created", "revived") or newly:
            return "raised"
        return "updated"

    # Nothing is held: clear what is standing, in the inbox that shows it.
    _raised.pop(endpoint_id, None)
    cleared = [clear(target, minutes) for target in inboxes]
    return "cleared" if any(cleared) else "none_held"
