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
#:
#: DEFAULT-ON UNDER ``bridge_consent_v2`` DOES NOT CHANGE THIS CONSTANT. The
#: registry literal and every consumer that reads ``enabled`` from disk (host,
#: read_context, authority, wake, engagement) keep reading an absent key as OFF,
#: so a hub that serves only v1 is byte-identical to the hold. The opt-out
#: default is a COMPOSED gate that needs a fact this module cannot know — the
#: hub's served consent — and so lives in
#: :meth:`TeamBridgeSettings.enabled_given`, used by the status discipline (the
#: only automatic enrolment) and the status route's ``local.enabled``. The
#: automatic enrolment then WRITES ``team_bridge.enabled = true`` once the owner
#: and receipt are committed (:func:`record_local_enabled_if_unset`), which is what makes
#: the offline consumers above agree; an absent key therefore never starts any
#: send/receive/wake work by itself — it only means "eligible to be enrolled".
DEFAULT_TEAM_BRIDGE_ENABLED = False

#: The one consent version whose copy discloses the opt-out default. Only a hub
#: that SERVES it (and lists it as accepted) can turn an absent
#: ``team_bridge.enabled`` into "eligible to be enrolled". Spelled here, not
#: imported from ``client``, because ``client`` imports this module.
DEFAULT_ON_CONSENT_VERSION = "bridge_consent_v2"

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
#: WHO WRITES IT. Under ``bridge_consent_v1`` nothing does (consent hold,
#: #2712: a status read must never enrol). Under ``bridge_consent_v2`` two
#: places do, and both mean "never enrol this account from this machine again":
#: the gated automatic enrolment (``StatusDiscipline._default_enroll``) after the
#: server created the endpoint, and the one-click turn-off (a successful
#: ``state=disabled`` write). It is MEMORY that suppresses a repeat, never
#: authority: no write is ever performed BECAUSE this key says so, and a missing
#: key never proves anything either (the server's own no-op-when-an-endpoint-
#: exists rule is the authoritative stickiness).
#:
#: ``pending:<endpoint_id>:<head>`` (D1, review R3/R2-1) is the one other
#: value: THIS machine's default-on POST created ``endpoint_id`` but the local
#: owner commit failed or was vetoed. Without it the account was marked
#: ``done`` from the hub's answer alone, the surface then read state C, and no
#: path ever bound the device ("on, but deaf"). The status discipline retries
#: ONLY the local commit against that exact endpoint — never a second POST —
#: seeded at ``head``, the feed head read BEFORE the original POST (``0`` when
#: there was none, a brand-new endpoint), so a message that arrived while the
#: device could not receive is never skipped. It still suppresses a repeat
#: POST exactly like ``done``; an explicit OFF or a Turn off overwrites it
#: with ``done``.
DEFAULT_TEAM_BRIDGE_ENROLLMENT: Mapping[str, str] = {}
DEFAULT_ENROLLMENT_DONE = "done"
DEFAULT_ENROLLMENT_PENDING_PREFIX = "pending:"


def pending_default_enrollment(endpoint_id: str, head: int | None) -> str:
    """The ``pending:<endpoint_id>:<head>`` marker value (see above)."""

    return f"{DEFAULT_ENROLLMENT_PENDING_PREFIX}{endpoint_id}:{head if head is not None else 0}"


def parse_pending_default_enrollment(value: Any) -> tuple[str, int] | None:
    """``(endpoint_id, pre-POST head)`` from a pending marker, else ``None``.

    Anything malformed reads as "no pending commit": the retry is a recovery,
    and a hand-edited value must never point it at an endpoint or a cursor.
    """

    if not isinstance(value, str) or not value.startswith(DEFAULT_ENROLLMENT_PENDING_PREFIX):
        return None
    endpoint_id, sep, head = value[len(DEFAULT_ENROLLMENT_PENDING_PREFIX) :].rpartition(":")
    if not sep or not endpoint_id or not head.isdigit() or (len(head) > 1 and head[0] == "0"):
        return None
    return endpoint_id, int(head)


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
        account: state
        for account, state in value.items()
        if isinstance(account, str)
        and account.strip()
        and (
            state == DEFAULT_ENROLLMENT_DONE or parse_pending_default_enrollment(state) is not None
        )
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
    #: Whether ``team_bridge.enabled`` was PRESENT in the document (any value,
    #: including a hand-edited ``null``, which reads OFF). It is what separates
    #: "the user said so" from "nobody said anything", the one distinction the
    #: default-on gate needs: an explicit value always wins over a default.
    enabled_is_set: bool = False

    def enabled_given(self, default_on_in_force: bool) -> bool:
        """The composed local switch: explicit value, else the served default.

        An explicit ``true`` or ``false`` (or an unreadable present value, which
        reads OFF) is returned as is. Only an ABSENT key consults the hub: it
        reads ON while ``default_on_in_force`` (the hub serves
        ``bridge_consent_v2`` and lists it as accepted) and OFF otherwise, so a
        v1-only hub sees exactly what :attr:`enabled` says today. ON here means
        "eligible to be enrolled", never "authorised": every send, receive and
        wake path still needs the committed owner and a current receipt.
        """

        if self.enabled_is_set:
            return self.enabled
        return bool(default_on_in_force)

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
        """The one place that says how the keys are typed and defaulted."""

        raw_enabled = get(ENABLED_PATH, _MISSING)
        return TeamBridgeSettings(
            enabled=_enabled(raw_enabled),
            enabled_is_set=raw_enabled is not _MISSING,
            onboarding=_onboarding(get(ONBOARDING_PATH, DEFAULT_TEAM_BRIDGE_ONBOARDING)),
            session=_session(get(SESSION_PATH, DEFAULT_TEAM_BRIDGE_SESSION)),
            default_enrollment=_default_enrollment(
                get(DEFAULT_ENROLLMENT_PATH, DEFAULT_TEAM_BRIDGE_ENROLLMENT)
            ),
        )


def record_default_enrollment(
    config_dir: "Path", account_id: str, value: str = DEFAULT_ENROLLMENT_DONE
) -> None:
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
    if value != DEFAULT_ENROLLMENT_DONE and parse_pending_default_enrollment(value) is None:
        raise ValueError("value must be 'done' or a pending:<endpoint_id>:<head> marker")

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
        # ``done`` is terminal: a pending marker landing after a Turn off (which
        # writes ``done``) must never reopen the retry it closed.
        if current.get(account_id) in (DEFAULT_ENROLLMENT_DONE, value):
            return
        current[account_id] = value
        settings_io._store(manager, DEFAULT_ENROLLMENT_PATH, current)


def _store_internal(config_dir: "Path", path: tuple[str, ...], value: Any) -> None:
    """Write one internal ``team_bridge.*`` leaf under the config write lock.

    The same guarded, merge-preserving path :func:`record_default_enrollment`
    uses (see its docstring for the lock, the pre-write readability guard and
    the function-local imports); kept separate because these writers store a
    scalar, not a read-merge-write mapping.
    """

    from local_operator import settings_io
    from local_operator.config import CONFIG_FILE_NAME, ConfigManager
    from local_operator.config_writer import config_write_lock

    with config_write_lock(Path(config_dir)):
        settings_io.require_readable_config_path(Path(config_dir) / CONFIG_FILE_NAME)
        settings_io._store(ConfigManager(config_dir), path, value)


def record_local_enabled(config_dir: "Path", value: bool) -> None:
    """Persist an EXPLICIT ``team_bridge.enabled`` from the runtime.

    The one-click turn-off (``False``, the sticky local opt-out every fence and
    the enrolment gate read) writes it after the server answered. The automatic
    enrolment does NOT use this for ``True``: it must never overwrite an
    explicit value, see :func:`record_local_enabled_if_unset`. Best-effort by
    contract at the call sites.
    """

    if not isinstance(value, bool):
        raise ValueError("value must be a bool")
    _store_internal(config_dir, ENABLED_PATH, value)


def record_local_enabled_if_unset(config_dir: "Path") -> bool:
    """Write ``team_bridge.enabled = true`` ONLY while the key is still absent.

    WHY NOT :func:`record_local_enabled`. The automatic enrolment decides to
    write after an awaited hub round trip; an explicit value (``false`` from a
    settings edit or the one-click Turn off, or an explicit ``true``) may have
    landed in that window, and "explicit false always wins" means the default
    must never overwrite it. The check and the write share ONE config write lock
    (reentrant with the leaf write), so no writer can slip in between them.
    Returns whether it wrote. Best-effort at the call site, like its sibling.
    """

    from local_operator import settings_io
    from local_operator.config import CONFIG_FILE_NAME, ConfigManager
    from local_operator.config_writer import config_write_lock

    with config_write_lock(Path(config_dir)):
        settings_io.require_readable_config_path(Path(config_dir) / CONFIG_FILE_NAME)
        manager = ConfigManager(config_dir)
        if TeamBridgeSettings.from_config(manager).enabled_is_set:
            return False
        settings_io._store(manager, ENABLED_PATH, True)
        return True


def record_local_session_if_unset(config_dir: "Path") -> bool:
    """Designate the account's chief-of-staff session ONLY while none is chosen.

    WHY (operator incident, 2026-10-28). A teammate's bridge delivered to a task
    session ("Hyundai POC") instead of her chief of staff: whatever session was
    on screen when Team Bridge was switched on became the wake destination, and
    nothing on the enrolment paths ever looked up the chief of staff. The
    designation is the wake consent (§13.6), so it must name a session the
    person would expect to be woken — and the one durable answer to "who is my
    chief of staff on this machine" is Aida's own record
    (:func:`local_operator.aida.state.session_id_of`), which survives renames.

    THE RULE. Write ``team_bridge.session`` = that session id when the key is
    ABSENT, under the config write lock, and never over an explicit choice: a
    person who picked a session in Settings keeps it. A person who CLEARED it is
    also an explicit choice, and that is why the key is stored empty rather than
    deleted (``settings_io``'s ``empty_unsets=False``): "absent" can only then
    mean "never touched", so a withdrawn wake consent is never re-granted by a
    later enable. A key deleted by an older build (before that change) is
    indistinguishable from one never set; the next enable defaults it once, which
    is the same default a fresh machine gets. It writes the SETTING, not the
    store, because the status read's ``_sync_designation`` mirrors the setting
    into the native record and
    would clear a store-only value on the next read. No Aida session on this
    machine → nothing is written (badge only, as before). Returns whether it
    wrote; best-effort at the call sites, like its siblings.
    """

    from local_operator import settings_io
    from local_operator.aida.state import session_id_of
    from local_operator.config import CONFIG_FILE_NAME, ConfigManager
    from local_operator.config_writer import config_write_lock

    chief = session_id_of(config_dir)
    if not chief:
        return False
    with config_write_lock(Path(config_dir)):
        settings_io.require_readable_config_path(Path(config_dir) / CONFIG_FILE_NAME)
        manager = ConfigManager(config_dir)
        # Presence matters: an explicit empty/null is badge-only consent, not
        # an invitation to choose a wake destination on the person's behalf.
        if manager.get_nested_value(SESSION_PATH, _MISSING) is not _MISSING:
            return False
        settings_io._store(manager, SESSION_PATH, chief)
        return True


def record_onboarding_seen(config_dir: "Path") -> None:
    """Mark the one-time notice dismissed on this machine (never un-set here)."""

    _store_internal(config_dir, ONBOARDING_PATH, ONBOARDING_SEEN)
