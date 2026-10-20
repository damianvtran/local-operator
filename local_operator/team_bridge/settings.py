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
(RFC §5.4), and every reader normalises an unknown value to the shipped
default (now ON/opt-out for ``enabled``), never to an invented state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.config import ConfigManager

#: ``team_bridge.enabled`` — ON by default (opt-out).
#:
#: The product requirement is that a signed-in account with an eligible Radient
#: Team plan has agent messages on without first walking an opt-in dialog; an
#: explicit ``false`` written here (the settings row, a hand edit) is that
#: machine's opt-out and wins over the default. This key is only the LOCAL half
#: of the composed gate — capability × eligible × enabled (RFC §5.4) — so the
#: default enables nothing on its own: ineligible, signed-out, dark-launched and
#: disabled-membership accounts are still refused by the server and the surface
#: matrix, and nothing (no boot path, no connector) may run while it is false.
DEFAULT_TEAM_BRIDGE_ENABLED = True

#: ``team_bridge.onboarding`` — the one-time dialog's shown-flag (RFC §5.3:
#: client-local in v1, deliberately never a server row). ``""`` means "not
#: dismissed yet"; :data:`ONBOARDING_SEEN` means the dialog was dismissed on
#: this machine. An unknown value normalises to ``""`` — for a UX flag the
#: safe direction is to allow the one-time notice to show again, never to
#: suppress it forever.
DEFAULT_TEAM_BRIDGE_ONBOARDING = ""
ONBOARDING_SEEN = "seen"

#: ``team_bridge.default_enrollment`` — internal state (deliberately NOT a
#: registry row, exactly like ``onboarding``): ``"done"`` records that this
#: machine's opt-out-default enrollment ran. It is the memory that keeps the
#: consent write to ONE: the server's endpoint is normally that memory, but an
#: endpoint can disappear (a server-side reset), and without this marker a
#: later status answer would read the same "eligible, no endpoint" state and
#: consent again on the user's behalf. Any value other than ``DONE`` reads as
#: "not enrolled by default" — the safe direction for a marker whose false
#: reading costs one idempotent ``active`` write, never a missed opt-out (the
#: local ``enabled`` key and the endpoint's own state are the opt-out gates).
DEFAULT_TEAM_BRIDGE_ENROLLMENT = ""
DEFAULT_ENROLLMENT_DONE = "done"

#: ``team_bridge.session`` — the designated CoS session id (RFC §5.6/§13.6).
#: The connector wakes ONE session per account, and the designation is the
#: consent for it to start a runtime; there is deliberately no list here,
#: because "which sessions may be woken" is not a user-tunable policy in v1
#: and a list would imply one. Empty = no destination; arrivals badge only and
#: never wake a session.
DEFAULT_TEAM_BRIDGE_SESSION = ""

ENABLED_PATH = ("team_bridge", "enabled")
ONBOARDING_PATH = ("team_bridge", "onboarding")
DEFAULT_ENROLLMENT_PATH = ("team_bridge", "default_enrollment")
SESSION_PATH = ("team_bridge", "session")


def _bool(value: Any, default: bool) -> bool:
    """A real bool, else the default (the ``hub_sync.settings._bool`` rule).

    A hand-edited ``"false"`` string must neither mean True (``bool("false")``)
    nor crash the reader: it is not a bool, so it reads as unset.
    """

    return value if isinstance(value, bool) else default


def _onboarding(value: Any) -> str:
    return ONBOARDING_SEEN if value == ONBOARDING_SEEN else DEFAULT_TEAM_BRIDGE_ONBOARDING


def _default_enrollment(value: Any) -> str:
    return DEFAULT_ENROLLMENT_DONE if value == DEFAULT_ENROLLMENT_DONE else ""


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

    Four: ``enabled`` and ``session`` are the registry's editable rows, and
    ``onboarding`` (the one-time dialog's shown-flag, §5.3) is internal state —
    deliberately NOT a registry row: it is UI bookkeeping, not a knob, and a
    public row would dress session-format state as configuration.
    ``default_enrollment`` is the same kind of internal state: the receipt that
    the opt-out default consented once, not a user preference. SCREENING IS
    NOT HERE AT ALL: it is the admission gate (§8, §13.9), an invariant the
    implementation enforces — not a setting any write can turn off.
    """

    enabled: bool = DEFAULT_TEAM_BRIDGE_ENABLED
    onboarding: str = DEFAULT_TEAM_BRIDGE_ONBOARDING
    session: str = DEFAULT_TEAM_BRIDGE_SESSION
    default_enrollment: str = DEFAULT_TEAM_BRIDGE_ENROLLMENT

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
        """The one place that says how the four keys are typed and defaulted."""

        return TeamBridgeSettings(
            enabled=_bool(
                get(ENABLED_PATH, DEFAULT_TEAM_BRIDGE_ENABLED), DEFAULT_TEAM_BRIDGE_ENABLED
            ),
            onboarding=_onboarding(get(ONBOARDING_PATH, DEFAULT_TEAM_BRIDGE_ONBOARDING)),
            session=_session(get(SESSION_PATH, DEFAULT_TEAM_BRIDGE_SESSION)),
            default_enrollment=_default_enrollment(
                get(DEFAULT_ENROLLMENT_PATH, DEFAULT_TEAM_BRIDGE_ENROLLMENT)
            ),
        )


def record_default_enrollment(config_dir: "Path") -> None:
    """Record that the opt-out default ran the consent write on this machine.

    Written through ``settings_io._store`` because this is INTERNAL state with
    no registry row (the same reason ``onboarding`` is hand-written): a public
    row would dress a one-time receipt as a knob. The import stays FUNCTION-LOCAL
    because ``settings_io`` is loaded on every CLI start and this module is on
    that path (see the module docstring); ``_store`` is the one merge-preserving
    nested writer, so a concurrent hand edit to a sibling ``team_bridge`` key is
    not clobbered by this write.

    Best-effort by contract: the caller (the status discipline) owns the
    enrollment itself and must not fail a status read because a bookkeeping
    write could not land.
    """

    from local_operator import settings_io
    from local_operator.config import CONFIG_FILE_NAME, ConfigManager

    # The SAME pre-write guard every settings write uses: an unreadable document
    # must abort before ConfigManager's recovery rename moves the user's file
    # aside. A missing file is a first run and is allowed.
    settings_io.require_readable_config_path(Path(config_dir) / CONFIG_FILE_NAME)
    settings_io._store(ConfigManager(config_dir), DEFAULT_ENROLLMENT_PATH, DEFAULT_ENROLLMENT_DONE)
