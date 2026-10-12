"""The ``team_bridge.*`` configuration keys, read the way the ``/settings`` registry writes them.

WHY THE DEFAULTS LIVE HERE. ``settings_io`` keeps this package off its import
path (the registry is loaded on every CLI start and on every ``lop serve``
boot), so the registry carries LITERAL defaults and
``tests/unit/test_settings_io.py::_consumer_defaults`` imports THESE constants
to pin that the two cannot drift. The readers sit next to the constants for the
same reason: one place says what "unset" means.

WHY ``get_nested_value``. Every key here is a genuinely nested tuple
(``("team_bridge", "enabled")``); the flat ``ConfigManager`` accessor looks a
dotted string up as ONE literal top-level key (the ``display.*`` exception),
so passing it ``team_bridge.enabled`` would silently read nothing and the
feature would look configured while never following the setting.

SCOPE. These are the LOCAL, per-machine keys. Eligibility, consent and the
endpoint state are SERVER state (``GET /v1/messaging/status`` and
``/settings``); nothing here duplicates them. The master switch is a machine
gate — the composed gate a surface checks is capability × eligible × enabled
(RFC §5.4), and every reader normalises fail-closed: an unknown value reads as
the shipped default, never as "on".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.config import ConfigManager

#: ``team_bridge.enabled`` — off by default. The feature ships dark on every
#: machine until a user opts in locally AND the server says they are eligible
#: and consented; this key is the local half, and nothing (no boot path, no
#: connector) may run while it is false.
DEFAULT_TEAM_BRIDGE_ENABLED = False

#: ``team_bridge.onboarding`` — the one-time dialog's shown-flag (RFC §5.3:
#: client-local in v1, deliberately never a server row). ``""`` means "not
#: dismissed yet"; :data:`ONBOARDING_SEEN` means the dialog was dismissed on
#: this machine. An unknown value normalises to ``""`` — for a UX flag the
#: safe direction is to allow the one-time notice to show again, never to
#: suppress it forever.
DEFAULT_TEAM_BRIDGE_ONBOARDING = ""
ONBOARDING_SEEN = "seen"

#: ``team_bridge.session`` — the designated CoS session id (RFC §5.6/§13.6).
#: The connector wakes ONE session per account, and the designation is the
#: consent for it to start a runtime; there is deliberately no list here,
#: because "which sessions may be woken" is not a user-tunable policy in v1
#: and a list would imply one. Empty = no destination; arrivals badge only and
#: never wake a session.
DEFAULT_TEAM_BRIDGE_SESSION = ""

ENABLED_PATH = ("team_bridge", "enabled")
ONBOARDING_PATH = ("team_bridge", "onboarding")
SESSION_PATH = ("team_bridge", "session")


def _bool(value: Any, default: bool) -> bool:
    """A real bool, else the default (the ``hub_sync.settings._bool`` rule).

    A hand-edited ``"false"`` string must neither mean True (``bool("false")``)
    nor crash the reader: it is not a bool, so it reads as unset.
    """

    return value if isinstance(value, bool) else default


def _onboarding(value: Any) -> str:
    return ONBOARDING_SEEN if value == ONBOARDING_SEEN else DEFAULT_TEAM_BRIDGE_ONBOARDING


def _session(value: Any) -> str:
    """One designated session id; anything else reads as "none designated".

    A non-string, or a list left over from an early hand-edit of this key's
    draft shape, reads as unset rather than being coerced: wrongly naming a
    wake destination is worse than naming none (badge-only is the safe state).
    """

    if isinstance(value, str):
        return value.strip()
    return DEFAULT_TEAM_BRIDGE_SESSION


@dataclass(frozen=True)
class TeamBridgeSettings:
    """The local keys, defaulted and normalised.

    Three: ``enabled`` and ``session`` are the registry's editable rows, and
    ``onboarding`` is internal state (the one-time dialog's shown-flag, §5.3 —
    deliberately NOT a registry row: it is UI bookkeeping, not a knob, and a
    public row would dress session-format state as configuration). SCREENING IS
    NOT HERE AT ALL: it is the admission gate (§8, §13.9), an invariant the
    implementation enforces — not a setting any write can turn off.
    """

    enabled: bool = DEFAULT_TEAM_BRIDGE_ENABLED
    onboarding: str = DEFAULT_TEAM_BRIDGE_ONBOARDING
    session: str = DEFAULT_TEAM_BRIDGE_SESSION

    @staticmethod
    def from_config(cm: "ConfigManager") -> "TeamBridgeSettings":
        return TeamBridgeSettings._derive(cm.get_nested_value)

    @staticmethod
    def from_values(values: Mapping[str, Any]) -> "TeamBridgeSettings":
        """Same derivation over a bare ``values`` mapping (a config document)."""

        def get(path: tuple[str, ...], default: Any) -> Any:
            node: Any = values
            for step in path:
                if not isinstance(node, Mapping) or step not in node:
                    return default
                node = node[step]
            return node

        return TeamBridgeSettings._derive(get)

    @staticmethod
    def _derive(get: Callable[..., Any]) -> "TeamBridgeSettings":
        """The one place that says how the three keys are typed and defaulted."""

        return TeamBridgeSettings(
            enabled=_bool(
                get(ENABLED_PATH, DEFAULT_TEAM_BRIDGE_ENABLED), DEFAULT_TEAM_BRIDGE_ENABLED
            ),
            onboarding=_onboarding(get(ONBOARDING_PATH, DEFAULT_TEAM_BRIDGE_ONBOARDING)),
            session=_session(get(SESSION_PATH, DEFAULT_TEAM_BRIDGE_SESSION)),
        )
