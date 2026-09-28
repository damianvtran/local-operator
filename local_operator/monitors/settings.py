"""``values.monitor.*`` — the settings snapshot one session's scheduler reads.

Every key the contract lists (design §16) gets a module-level default constant
HERE, beside the one reader that applies them, and
``tests/unit/test_settings_io.py::_monitor_consumer_defaults`` binds the
registry rows to these constants so a default cannot drift between the page and
the consumer.

The section is scoped ``NEW_SESSIONS`` in ``settings_io`` for the classification
section's reason: the scheduler is built once per session and handed a SNAPSHOT
of this section, so an edit lands on the next session start. Claiming LIVE
would be a painted lie.

Reading never raises: a malformed config must not stop a session from starting
(the ``_configured_max_running`` posture). An unusable value falls back to the
default with a warning, EXCEPT where the contract makes a small value
meaningful (a non-positive interval/timeout is refused the same way, because 0
would schedule a hot loop or park every check on a deadline that can never
fire).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from local_operator.settings_io import strict_bool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The defaults — one per §16 key, each named for its consumer
# ---------------------------------------------------------------------------

#: ``defaultIntervalS`` — the interval when the tool's ``every`` is omitted.
#: Read by the scheduler's create flow; the 30 s floor is the code constant
#: ``spec.MIN_MONITOR_INTERVAL_MS``, deliberately NOT a setting.
DEFAULT_DEFAULT_INTERVAL_S = 60

#: ``maxMonitors`` — per-session cap (§11.4). Read by the scheduler's create
#: flow; the rejection sentence names this number.
DEFAULT_MAX_MONITORS = 8

#: ``runTimeoutMs`` — per-check deadline (§5.4). Read by the scheduler, which
#: hands it to the session's check runner.
DEFAULT_RUN_TIMEOUT_MS = 120_000

#: ``snapshotMaxChars`` — stored normalized-snapshot cap (§10.3). Read by the
#: diff module through the scheduler; bounds the per-monitor disk footprint and
#: the diff input.
DEFAULT_SNAPSHOT_MAX_CHARS = 32_768

#: ``maxDeltaLines`` — changed lines summarised per delivery (§7.3).
DEFAULT_MAX_DELTA_LINES = 12

#: ``deltaMaxChars`` — total delta text, per delivery message (§7.3).
DEFAULT_DELTA_MAX_CHARS = 1_200

#: ``classifyMaxChars`` — the classifier state bound (§8). Registered in slice
#: 1 so the key exists for the next session; SLICE 2 (the classifier gate) is
#: its consumer. Kept here rather than in a slice-2 module so the registry
#: binding has one home from the day the key ships.
DEFAULT_CLASSIFY_MAX_CHARS = 1_200

#: ``maxConsecutiveFailures`` — auto-disable threshold (§11.3). Read by the
#: scheduler's failure ladder.
DEFAULT_MAX_CONSECUTIVE_FAILURES = 5

#: ``maxDeliveriesPerHour`` — per-monitor delivery cap (§9.4). Read by the
#: scheduler's rate window.
DEFAULT_MAX_DELIVERIES_PER_HOUR = 12

#: ``normalizeTimestamps`` — strip timestamp churn before diffing (§7.1).
#: Read by the diff module through the scheduler.
DEFAULT_NORMALIZE_TIMESTAMPS = True


@dataclass(frozen=True)
class MonitorSettings:
    """A validated snapshot of ``values.monitor``, taken once per session."""

    default_interval_s: int = DEFAULT_DEFAULT_INTERVAL_S
    max_monitors: int = DEFAULT_MAX_MONITORS
    run_timeout_ms: int = DEFAULT_RUN_TIMEOUT_MS
    snapshot_max_chars: int = DEFAULT_SNAPSHOT_MAX_CHARS
    max_delta_lines: int = DEFAULT_MAX_DELTA_LINES
    delta_max_chars: int = DEFAULT_DELTA_MAX_CHARS
    classify_max_chars: int = DEFAULT_CLASSIFY_MAX_CHARS
    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES
    max_deliveries_per_hour: int = DEFAULT_MAX_DELIVERIES_PER_HOUR
    normalize_timestamps: bool = DEFAULT_NORMALIZE_TIMESTAMPS

    @property
    def default_interval_ms(self) -> int:
        return self.default_interval_s * 1000

    @classmethod
    def from_values(cls, values: Mapping[str, Any] | None) -> MonitorSettings:
        """Build from a ``values.monitor`` mapping; every defect falls back.

        ``0`` (and any non-positive number) means "use the default" for the
        numeric keys, matching the classification section's spelling: the
        readers refuse a non-positive value rather than honouring it, so a
        hand-edit cannot leave a 0-second poll or a zero-length deadline.
        """
        if not isinstance(values, Mapping):
            return cls()

        def as_int(key: str, default: int) -> int:
            raw = values.get(key)
            if raw is None:
                return default
            try:
                parsed = int(raw)
            except (TypeError, ValueError):
                logger.warning("monitor.%s=%r is not a number; using %d", key, raw, default)
                return default
            if parsed <= 0:
                logger.warning("monitor.%s=%r must be positive; using %d", key, raw, default)
                return default
            return parsed

        return cls(
            default_interval_s=as_int("defaultIntervalS", DEFAULT_DEFAULT_INTERVAL_S),
            max_monitors=as_int("maxMonitors", DEFAULT_MAX_MONITORS),
            run_timeout_ms=as_int("runTimeoutMs", DEFAULT_RUN_TIMEOUT_MS),
            snapshot_max_chars=as_int("snapshotMaxChars", DEFAULT_SNAPSHOT_MAX_CHARS),
            max_delta_lines=as_int("maxDeltaLines", DEFAULT_MAX_DELTA_LINES),
            delta_max_chars=as_int("deltaMaxChars", DEFAULT_DELTA_MAX_CHARS),
            classify_max_chars=as_int("classifyMaxChars", DEFAULT_CLASSIFY_MAX_CHARS),
            max_consecutive_failures=as_int(
                "maxConsecutiveFailures", DEFAULT_MAX_CONSECUTIVE_FAILURES
            ),
            max_deliveries_per_hour=as_int("maxDeliveriesPerHour", DEFAULT_MAX_DELIVERIES_PER_HOUR),
            normalize_timestamps=strict_bool(
                values.get("normalizeTimestamps"), DEFAULT_NORMALIZE_TIMESTAMPS
            ),
        )


def read_monitor_settings(values: Mapping[str, Any] | None = None) -> MonitorSettings:
    """Read ``values.monitor`` from config, or from an already-read mapping.

    ``None`` reads the file through a fresh ``ConfigManager`` — the same shape
    ``Session`` uses for ``values.subagents.max_running``. Never raises: a bad
    config must not fail session startup.
    """
    try:
        if values is None:
            from local_operator.config import ConfigManager
            from local_operator.paths import config_dir

            values = ConfigManager(config_dir()).get_config_value("monitor", None)
        return MonitorSettings.from_values(values if isinstance(values, Mapping) else None)
    except Exception:  # noqa: BLE001 — a bad config must not fail session startup
        logger.warning("values.monitor could not be read; using the built-in defaults")
        return MonitorSettings()
