"""The ``hub.org.auto_enrol`` setting, read the way the ``/settings`` registry writes it.

WHY THE DEFAULT LIVES HERE. ``settings_io`` keeps the hub stack off its import
path, so the registry carries a literal default and
``tests/unit/test_settings_io.py::_consumer_defaults`` imports THIS constant to
pin that the two cannot drift (AGENTS.md "Adding a configuration key").

WHY ON BY DEFAULT. The operator asked that agents "pull the org's latest teams,
agents and skills and subscribe automatically to stay up to date". Auto-enrol
only ever runs for a Radient Team member (:mod:`.eligibility`), only adds items
the machine does not have (conflicts are skipped, never overwritten) and never
auto-pushes, so the default costs an ineligible user nothing and an eligible
one gets the catalogue without asking. Turning it off stops new-item pulls
only; already-linked items keep following ``hub.auto_update.*``.

The path is a genuinely NESTED tuple read through ``get_nested_value`` /
``read_config_values`` — never the flat dotted accessor, which would look up a
literal top-level key nothing writes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

DEFAULT_ORG_AUTO_ENROL = True
AUTO_ENROL_PATH = ("hub", "org", "auto_enrol")


def _from_values(values: Mapping[str, Any] | None) -> bool:
    current: Any = values
    for part in AUTO_ENROL_PATH:
        if not isinstance(current, Mapping) or part not in current:
            return DEFAULT_ORG_AUTO_ENROL
        current = current[part]
    # A hand-edited "false" string is not a bool: it reads as unset rather than
    # as True (``bool("false")``).
    return current if isinstance(current, bool) else DEFAULT_ORG_AUTO_ENROL


def auto_enrol_enabled(config_dir: Path) -> bool:
    """The setting as it is on disk NOW (a LIVE key: no relaunch needed)."""

    try:
        from local_operator.config import read_config_values

        return _from_values(read_config_values(Path(config_dir)))
    except Exception:  # noqa: BLE001 - a torn/unreadable config reads as the default
        return DEFAULT_ORG_AUTO_ENROL
