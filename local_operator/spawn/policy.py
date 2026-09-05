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

#: Forking follows the branch in this terminal; the owner retains the original
#: work. Explicit window preferences remain honored, without a config migration.
DEFAULT_FORK_MODE = "switch"

#: Under cmux, a fork gets its own sidebar workspace by default rather than a
#: tab in the current one. The workspace form is also the one that needs a single
#: CLI call (``new-workspace`` carries the command; ``new-surface`` does not),
#: so there is no window in which the fork exists but is not yet running.
DEFAULT_FORK_CMUX_PLACEMENT = "workspace"

#: Valid ``fork.mode`` values, mirroring the registry's choices.
FORK_MODE_WINDOW = "window"
FORK_MODE_SWITCH = "switch"


def parse_fork_args(arg: str) -> tuple[str | None, str]:
    """Parse leading destination flags without rewriting the prompt's quoting.

    Only leading tokens are options. `--` protects a literal flag-looking
    opening instruction, and the rest stays byte-for-byte text rather than
    round-tripping through shlex (the model, not a shell, receives it).
    """
    mode = None
    rest = arg.strip()
    while rest.startswith("--"):
        # split(None, 1) also admits tabs/newlines between options and prose.
        parts = rest.split(None, 1)
        flag = parts[0]
        remainder = parts[1] if len(parts) > 1 else ""
        if flag == "--":
            return mode, remainder
        if flag not in ("--window", "--switch"):
            raise ValueError(
                f"unknown fork option {flag}; use --window, --switch, or -- before text"
            )
        selected = flag[2:]
        if mode is not None and mode != selected:
            raise ValueError("choose either --window or --switch, not both")
        mode = selected
        rest = remainder
    return mode, rest


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
