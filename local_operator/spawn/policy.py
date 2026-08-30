"""How ``/fork`` is configured, and the defaults that are the single source.

These constants are the CONSUMER the settings registry's anti-drift test
compares against (``tests/unit/test_settings_io.py::_consumer_defaults``). That
test exists because a registry default that disagrees with the code's default is
a painted lie: the settings page shows one value and the feature uses another,
and nothing reports it. Keeping the defaults here — where the feature reads them
— and mapping them there is what makes the two provably the same.
"""

from __future__ import annotations

from typing import Any, Mapping

#: Where a fork opens. ``window`` keeps the current session running and opens the
#: branch elsewhere, which is the whole point of forking rather than switching.
DEFAULT_FORK_MODE = "window"

#: Under cmux, a fork gets its own sidebar workspace by default rather than a
#: tab in the current one. The workspace form is also the one that needs a single
#: CLI call (``new-workspace`` carries the command; ``new-surface`` does not),
#: so there is no window in which the fork exists but is not yet running.
DEFAULT_FORK_CMUX_PLACEMENT = "workspace"

#: Valid ``fork.mode`` values, mirroring the registry's choices.
FORK_MODE_WINDOW = "window"
FORK_MODE_SWITCH = "switch"


def fork_mode(values: Mapping[str, Any] | None) -> str:
    """``fork.mode`` from a config mapping, falling back to the default.

    Reads the NESTED path (``values["fork"]["mode"]``), which is what the
    settings registry declares and therefore what ``lop config edit`` and the
    ``/settings`` page write. An unknown value degrades to the default rather
    than raising: a typo in config.yml should cost the non-default behaviour,
    never the fork.
    """
    raw = _nested(values, "fork", "mode")
    return raw if raw in (FORK_MODE_WINDOW, FORK_MODE_SWITCH) else DEFAULT_FORK_MODE


def fork_cmux_placement(values: Mapping[str, Any] | None) -> str:
    """``fork.cmux_placement`` from a config mapping, or the default."""
    from local_operator.spawn.cmux import PLACEMENT_SURFACE, PLACEMENT_WORKSPACE

    raw = _nested(values, "fork", "cmux_placement")
    if raw in (PLACEMENT_WORKSPACE, PLACEMENT_SURFACE):
        return raw
    return DEFAULT_FORK_CMUX_PLACEMENT


def _nested(values: Mapping[str, Any] | None, *path: str) -> str:
    """Walk a nested config path, returning ``""`` for anything unexpected.

    Tolerant by design: this reads a user-editable YAML file, where a hand-typed
    ``fork: switch`` (a string where a mapping belongs) is an ordinary mistake
    and must not raise on the command path.
    """
    current: Any = values
    for key in path:
        if not isinstance(current, Mapping):
            return ""
        current = current.get(key)
    return current if isinstance(current, str) else ""
