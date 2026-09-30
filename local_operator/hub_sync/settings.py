"""Settings for hub auto-update, read the way the ``/settings`` registry writes them.

WHY THE DEFAULTS LIVE HERE. ``settings_io`` keeps hub-sync off its import path
(it is loaded on every CLI start), so the registry carries literal defaults and
``tests/unit/test_settings_io.py::_consumer_defaults`` imports THESE constants to
pin that the two cannot drift. The reader sits next to the constants for the
same reason the other consumers do: one place says what "unset" means.

WHY ``get_nested_value``. The four keys are genuinely nested tuples
(``("hub", "auto_update", "agents")``). the flat ``ConfigManager`` accessor looks
a dotted string up as ONE literal top-level key (the ``display.*`` exception),
so passing it ``hub.auto_update.agents`` silently reads nothing and auto-update
would look configured while never following the setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.config import ConfigManager

DEFAULT_AUTO_UPDATE_AGENTS = True
DEFAULT_AUTO_UPDATE_TEAMS = True
DEFAULT_CHECK_INTERVAL_MIN = 60
MIN_CHECK_INTERVAL_MIN = 5
MAX_CHECK_INTERVAL_MIN = 1440

AUTO_AGENTS_PATH = ("hub", "auto_update", "agents")
AUTO_TEAMS_PATH = ("hub", "auto_update", "teams")
INTERVAL_PATH = ("hub", "check_interval_min")
MERGE_MODEL_PATH = ("hub", "merge_model")


def _bool(value: Any, default: bool) -> bool:
    """A real bool, else the default.

    A hand-edited ``"false"`` string must neither mean True (``bool("false")``)
    nor crash: it is not a bool, so it reads as unset.
    """

    return value if isinstance(value, bool) else default


def _interval(value: Any) -> int:
    # ``bool`` is an ``int`` subclass; ``True`` is not a number of minutes.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return DEFAULT_CHECK_INTERVAL_MIN
    return max(MIN_CHECK_INTERVAL_MIN, min(MAX_CHECK_INTERVAL_MIN, int(value)))


@dataclass(frozen=True)
class HubSyncSettings:
    """The four ``hub.*`` keys, defaulted and clamped."""

    auto_agents: bool = DEFAULT_AUTO_UPDATE_AGENTS
    auto_teams: bool = DEFAULT_AUTO_UPDATE_TEAMS
    interval_min: int = DEFAULT_CHECK_INTERVAL_MIN
    merge_model: str = ""

    @staticmethod
    def _derive(get: Callable[..., Any]) -> "HubSyncSettings":
        """The one place that says how the four keys are typed, defaulted and clamped."""

        merge_model = get(MERGE_MODEL_PATH, "")
        return HubSyncSettings(
            auto_agents=_bool(get(AUTO_AGENTS_PATH), DEFAULT_AUTO_UPDATE_AGENTS),
            auto_teams=_bool(get(AUTO_TEAMS_PATH), DEFAULT_AUTO_UPDATE_TEAMS),
            interval_min=_interval(get(INTERVAL_PATH)),
            merge_model=merge_model.strip() if isinstance(merge_model, str) else "",
        )

    @staticmethod
    def from_config(cm: "ConfigManager") -> "HubSyncSettings":
        return HubSyncSettings._derive(cm.get_nested_value)

    @staticmethod
    def from_values(values: Mapping[str, Any]) -> "HubSyncSettings":
        """Same derivation over a bare ``values`` mapping (see ``read_config_values``)."""

        def get(path: tuple[str, ...], default: Any = None) -> Any:
            current: Any = values
            for part in path:
                if not isinstance(current, Mapping) or part not in current:
                    return default
                current = current[part]
            return current

        return HubSyncSettings._derive(get)

    @staticmethod
    def read_fresh(cm: "ConfigManager") -> "HubSyncSettings":
        """The settings as they are on disk NOW, not as this process last loaded them.

        The section is LIVE (``settings_io``): an edit from the TUI, the CLI or
        another daemon must land within one tick. A long-lived ``ConfigManager``
        (the server's) holds the config it read at boot, so a write made by a
        different process would never reach it. Reads the file directly
        (``read_config_values``) instead of constructing a second manager: this
        runs from a timer and from every context, and construction can move an
        unparseable ``config.yml`` aside. Falls back to the given manager when the
        file cannot be read right now (a torn write) or the manager has no directory.
        """

        config_dir = getattr(cm, "config_dir", None)
        if config_dir is not None:
            try:
                from local_operator.config import read_config_values

                values = read_config_values(config_dir)
            except Exception:  # noqa: BLE001 - never lose the tick over a settings read
                values = None
            if values is not None:
                return HubSyncSettings.from_values(values)
        return HubSyncSettings.from_config(cm)

    def auto_for(self, kind: str) -> bool:
        return self.auto_teams if kind == "team" else self.auto_agents
