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
shipped default (OFF for ``enabled`` during the consent hold), while a PRESENT
but unparseable value reads what the shipped default says, never a coerced ON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.config import ConfigManager

#: ``team_bridge.enabled`` — OFF by default (a consent hold).
#:
#: #2712 shipped the opt-out default ON, but the server still served
#: ``bridge_consent_v1``, whose copy describes the data handling without ever
#: offering an opt-out. A status read could therefore turn an eligible Team
#: plan on and write a consent receipt before the member saw anything that
#: asked them. Until a consent version carrying opt-out copy is served AND
#: accepted, an absent key must mean OFF. This key remains only the LOCAL half
#: of the composed gate — capability × eligible × enabled (RFC §5.4) — so an
#: explicit ``true`` (the settings row, a hand edit) enables only the surfaces on
#: this machine; the server still decides eligibility, consent and launch.
#: Restoring the ON default belongs to the new-consent-version change, not a
#: local flip.
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
#: ``"done"`` recording the consent write an earlier opt-out default ran for
#: that member. PER ACCOUNT on purpose: it was a per-member product rule, so
#: account A enrolling did not suppress account B on the same machine.
#:
#: UNDER THE CONSENT HOLD NOTHING WRITES THIS. No status read may perform a
#: consent write (#2712's auto-enrollment), and the explicit settings path does
#: not need a per-account memory because the user's own PUT is the action. The
#: reader and writer stay so an older build's file still loads and so a future
#: consent version can reuse the shape; they must never drive auto-consent.
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
    an earlier opt-out default consented once, not a user preference; under the
    consent hold nothing writes it and it drives no consent. SCREENING IS
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
    """Record a completed default-enrollment consent write for ``account_id``.

    Retained through the consent hold: no production path calls this while the
    status read is forbidden from enrolling, but the stored marker keeps its
    reader and a future consent version may reuse the shape. It must never be
    what authorizes a consent write.

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
