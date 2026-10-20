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
(RFC §5.4). Each reader's fallback is deliberate: an ABSENT value reads the
shipped default (OFF for ``enabled`` until ``bridge_consent_v2`` is served and
accepted), while a PRESENT
but unparseable value reads what the shipped default says, never a coerced ON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.config import ConfigManager

#: ``team_bridge.enabled`` — OFF by default (the consent hold continues).
#:
#: #2712 shipped the opt-out default ON while the server still served
#: ``bridge_consent_v1``, whose copy described the data handling without ever
#: offering an opt-out. A status read could therefore turn an eligible Team
#: plan on and write a consent receipt before the member saw anything that
#: asked them. The hold keeps an absent key OFF. The server now has
#: ``bridge_consent_v2``, whose copy says on-by-default, but this build still
#: ships opt-in: an explicit ``true`` (the settings row, a hand edit) is the
#: member's enable. The automatic enrollment exists behind that explicit local
#: gate and refuses any served version except v2
#: (``client.DEFAULT_ENROLLMENT_CONSENT_VERSION``). Flipping this constant (and
#: its pinned registry literal) to True after v2 is approved is the single
#: product change: a v1-only server is still left untouched.
#:
#: This key remains only the LOCAL half of the composed gate — capability ×
#: eligible × enabled (RFC §5.4) — so an explicit ``true`` enables only the
#: surfaces on this machine; the server still decides eligibility, consent and
#: launch.
DEFAULT_TEAM_BRIDGE_ENABLED = False

#: ``team_bridge.onboarding`` — the one-time dialog's shown-flag (RFC §5.3:
#: client-local in v1, deliberately never a server row). ``""`` means "not
#: dismissed yet"; :data:`ONBOARDING_SEEN` means the dialog was dismissed on
#: this machine. An unknown value normalises to ``""`` — for a UX flag the
#: safe direction is to allow the one-time notice to show again, never to
#: suppress it forever.
DEFAULT_TEAM_BRIDGE_ONBOARDING = ""
ONBOARDING_SEEN = "seen"

#: ``team_bridge.default_enrollment`` — internal state (deliberately NOT a
#: registry row, exactly like ``onboarding``): a mapping of ACCOUNT id to
#: ``"done"`` recording that the v2 automatic enrollment's consent write ran
#: for that member. PER ACCOUNT on purpose: the default is a per-member product
#: rule, so account A enrolling must not suppress account B on the same machine.
#: It is the memory that keeps each consent write to ONE: the server's endpoint
#: is normally that memory, but an endpoint can disappear (a server-side reset),
#: and without this marker a later status answer would read the same "eligible,
#: no endpoint" state and consent again on the user's behalf. Only a literal
#: ``"done"`` under a non-empty account id counts; anything else reads as "not
#: enrolled by default", the safe direction for a marker whose only job is to
#: suppress a REPEAT of a write the member's local enable already authorised.
#: The marker is never what AUTHORISES a write: the v2 served version and the
#: local ``enabled`` key are.
DEFAULT_TEAM_BRIDGE_ENROLLMENT: Mapping[str, str] = {}
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


#: Distinguishes "the key is absent" (the shipped default) from "the key is
#: present and unreadable" (fail closed). ``None`` cannot carry that difference:
#: a hand-edited ``team_bridge: {enabled: null}`` must not read as absent.
_MISSING = object()


def _enabled(value: Any) -> bool:
    """The shipped default, with a present-but-unreadable value reading OFF.

    WHY THIS IS NOT ``_bool(value, DEFAULT)``. ``None`` cannot tell an absent
    key from a hand-edited ``enabled: null``; only an ABSENT key may read the
    shipped default. A real bool is itself, and anything else fails closed —
    the same direction ``read_context`` already takes for a torn read.
    """

    if value is _MISSING:
        return DEFAULT_TEAM_BRIDGE_ENABLED
    return value if isinstance(value, bool) else False


def _onboarding(value: Any) -> str:
    return ONBOARDING_SEEN if value == ONBOARDING_SEEN else DEFAULT_TEAM_BRIDGE_ONBOARDING


def _default_enrollment(value: Any) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        account: DEFAULT_ENROLLMENT_DONE
        for account, state in value.items()
        if isinstance(account, str) and account.strip() and state == DEFAULT_ENROLLMENT_DONE
    }


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
    the v2 automatic enrollment consented once, not a user preference; it is the
    memory that suppresses a repeat, never the authority for a write. SCREENING IS
    NOT HERE AT ALL: it is the admission gate (§8, §13.9), an invariant the
    implementation enforces — not a setting any write can turn off.
    """

    enabled: bool = DEFAULT_TEAM_BRIDGE_ENABLED
    onboarding: str = DEFAULT_TEAM_BRIDGE_ONBOARDING
    session: str = DEFAULT_TEAM_BRIDGE_SESSION
    default_enrollment: Mapping[str, str] = field(default_factory=dict)

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
            enabled=_enabled(get(ENABLED_PATH, _MISSING)),
            onboarding=_onboarding(get(ONBOARDING_PATH, DEFAULT_TEAM_BRIDGE_ONBOARDING)),
            session=_session(get(SESSION_PATH, DEFAULT_TEAM_BRIDGE_SESSION)),
            default_enrollment=_default_enrollment(
                get(DEFAULT_ENROLLMENT_PATH, DEFAULT_TEAM_BRIDGE_ENROLLMENT)
            ),
        )


def record_default_enrollment(config_dir: "Path", account_id: str) -> None:
    """Record a completed v2 automatic-enrollment consent write for ``account_id``.

    Memory, never authority: the caller runs only after a successful
    ``bridge_consent_v2`` settings PUT, and the stored marker merely suppresses a
    repeat when a later status answer loses the endpoint. The v2 served version
    and the local ``enabled`` key are what authorise the write.

    Written through ``settings_io._store`` because this is INTERNAL state with
    no registry row (the same reason ``onboarding`` is hand-written): a public
    row would dress a one-time receipt as a knob. The import stays FUNCTION-LOCAL
    because ``settings_io`` is loaded on every CLI start and this module is on
    that path (see the module docstring); ``_store`` is the one merge-preserving
    nested writer, so a concurrent hand edit to a sibling ``team_bridge`` key is
    not clobbered by this write.

    Read-merge-write runs under the config write lock (not just the leaf write):
    two members enrolling on the same machine must both land in the mapping, and
    the second must not overwrite the first's entry. Best-effort by contract:
    the caller owns the enrollment itself and must not fail a status read
    because a bookkeeping write could not land.
    """

    if not isinstance(account_id, str) or not account_id.strip():
        raise ValueError("account_id must be a non-empty string")

    from local_operator import settings_io
    from local_operator.config import CONFIG_FILE_NAME, ConfigManager
    from local_operator.config_writer import config_write_lock

    with config_write_lock(Path(config_dir)):
        # The SAME pre-write guard every settings write uses: an unreadable
        # document must abort before ConfigManager's recovery rename moves the
        # user's file aside. A missing file is a first run and is allowed.
        settings_io.require_readable_config_path(Path(config_dir) / CONFIG_FILE_NAME)
        manager = ConfigManager(config_dir)
        current = dict(TeamBridgeSettings.from_config(manager).default_enrollment)
        if current.get(account_id) == DEFAULT_ENROLLMENT_DONE:
            return
        current[account_id] = DEFAULT_ENROLLMENT_DONE
        settings_io._store(manager, DEFAULT_ENROLLMENT_PATH, current)
