"""Notification routing by origin: which surface hears about a finished run.

WHY THIS EXISTS (operator, 2026-11-17, quoted in the mobile repo's
``docs/adr/0009-notification-routing-by-origin.md``): "if started on mobile it
should notify on mobile, if started on other channels/desktop/remote it should
notify on the desktop unless we can tell the user is not there and there's an
ask or something that requires the user's attention, then notify on mobile".

So every completion row now carries two facts the session stamps at run end
(``AgentEndEvent.origin``/``needs_user``, ``attention.db`` columns of the same
names), and this module is the ONE place that turns a row plus the paired
phones' preferences into "would the phone be told". Two callers must agree on
that answer, and a second hand-written copy is how they would drift (ADR 0009
§2 footnote 1):

- the push worker (``push_worker._is_candidate`` and ``_preference_gate``)
  decides whether to emit;
- the desktop banner ladder (``desktop_feed._focus_policy_for``,
  ``serving._announce_completion``) stays SILENT for a mobile-origin
  completion only while the phone would be told, so an item is never silent
  on every surface.

Pure functions of the row and the device registry (a read-only file read), so
the server process can call them. It lives in ``notifications`` rather than
``mobile`` because both surfaces read it, and the desktop feed is held by its
own guard to never reach into the mobile package (it composes and publishes;
it must not acquire or spawn); the registry read is a lazy import here.

The presence hold (ADR 0008 rules 1-5) is not decided here: it is a deferral,
not a routing decision (``push_worker._DeskGate``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

#: The run origin that routes to the phone. Every other value (``remote``,
#: ``desktop``, ``tui``, ``unknown`` and the legacy ``""``) routes as non-mobile,
#: which errs quiet on the phone (ADR 0009 R8).
ORIGIN_MOBILE = "mobile"

#: Kinds that reach any surface at all. ``interrupted``/``closed`` are a
#: deliberate stop and stay silent everywhere (ADR 0009 §2 matrix), which is
#: ``desktop_sessions.BRIDGE_NOTIFIABLE_KINDS``; repeated here as a literal to
#: keep this module importable without the server package, and held equal by
#: ``tests/unit/mobile/test_push_routing_origin.py``.
NOTIFIABLE_KINDS = frozenset({"complete", "error", "retired"})


def is_mobile_origin(state: Mapping[str, Any]) -> bool:
    return state.get("origin") == ORIGIN_MOBILE


def row_category(state: Mapping[str, Any]) -> str:
    """The preference category a row answers to.

    ``attention`` for a run that ended on an ask it opened (``needs_user``),
    whatever its kind: "Agent finished: Off" must never mute a question (ADR
    0009 §3 push-worker edit 4). Then ``errors`` for ``error``, and
    ``completions`` for everything else, as before.
    """
    if state.get("needs_user"):
        return "attention"
    return "errors" if state.get("kind") == "error" else "completions"


def device_wants_row(switches: Mapping[str, Any], state: Mapping[str, Any]) -> bool:
    """Whether ONE device's resolved preferences want this row pushed.

    ``switches`` is ``push_devices.resolved_preferences`` output, so every key
    is present. The completion rule is the scope: ``phone`` wants only what was
    started (driven) on the phone, ``all`` wants every completion.
    """
    if not switches.get("enabled", True):
        return False
    category = row_category(state)
    if category == "attention":
        return switches.get("attention", "away") != "never"
    if category == "errors":
        return bool(switches.get("errors", True))
    if not switches.get("completions", True):
        return False
    return is_mobile_origin(state) or switches.get("completions_scope", "phone") == "all"


def device_routes_row(switches: Mapping[str, Any], state: Mapping[str, Any]) -> bool:
    """The ROUTING half of :func:`device_wants_row`: scope and attention only.

    What the worker's candidacy gate asks (ADR 0009 §3 edit 1). The master
    switch and the category switches stay with the preference gate at
    composition, which records an explicit "no device wants it" outcome.
    """
    if state.get("needs_user"):
        return switches.get("attention", "away") != "never"
    if state.get("kind") == "error":
        return True
    return is_mobile_origin(state) or switches.get("completions_scope", "phone") == "all"


def any_device_routes_row(
    state: Mapping[str, Any],
    preferences: Mapping[str, Mapping[str, Any]],
    live: Iterable[str],
) -> bool:
    from local_operator.mobile import push_devices

    defaults = push_devices.PREFERENCE_DEFAULTS
    return any(device_routes_row(preferences.get(str(device), defaults), state) for device in live)


def any_device_wants_row(
    state: Mapping[str, Any],
    preferences: Mapping[str, Mapping[str, Any]],
    live: Iterable[str],
) -> bool:
    """Per COMPUTER: completion emits cannot target or exclude a device, so the
    most permissive live device wins (``push_worker._preference_gate``)."""
    from local_operator.mobile import push_devices

    defaults = push_devices.PREFERENCE_DEFAULTS
    return any(device_wants_row(preferences.get(str(device), defaults), state) for device in live)


def row_is_routable(state: Mapping[str, Any]) -> bool:
    """The ROW's own gates, before any device is asked.

    Unseen, a notifiable kind, and either ``needs_user`` (which bypasses
    ``notify``, ADR 0009 §3: ``notify`` is false for exactly the wake runs that
    end on an ask) or ``notify``. For completions and errors origin routing
    never turns a ``notify = false`` row back on.
    """
    if not state.get("unseen"):
        return False
    if state.get("kind") not in NOTIFIABLE_KINDS:
        return False
    return bool(state.get("needs_user")) or bool(state.get("notify", True))


def live_device_ids(config_dir: Path) -> list[str]:
    """Gate 4, the same read the worker's ``_live_device_ids`` makes."""
    from local_operator.mobile import push_devices

    try:
        facts = push_devices.credential_facts(config_dir)
    except (push_devices.PushRegistryCorrupt, OSError):
        return []
    return [
        str(fact["device_id"])
        for fact in facts
        if fact.get("state") == push_devices.STATE_LIVE and bool(fact.get("credential_live"))
    ]


def phone_would_emit(
    state: Mapping[str, Any],
    config_dir: Path,
    *,
    live: Iterable[str] | None = None,
) -> bool:
    """Would the phone be told about this row? (ADR 0009 §2 footnote 1.)

    False when no live device exists, when ``enabled`` or the row's category is
    off on every live device, or when the row's own gates refuse it. An
    unreadable registry answers False, which keeps the desktop banner on: the
    caller's only use of a False is "do not stay silent".
    """
    from local_operator.mobile import push_devices

    if not row_is_routable(state):
        return False
    devices = list(live) if live is not None else live_device_ids(config_dir)
    if not devices:
        return False
    try:
        preferences = push_devices.device_preferences(config_dir)
    except (push_devices.PushRegistryCorrupt, OSError):
        return False
    return any_device_wants_row(state, preferences, devices)


def desktop_defers_to_phone(state: Mapping[str, Any], config_dir: Path) -> bool:
    """Whether the DESKTOP banner stays silent for this row (ADR 0009 §2, ¹).

    Only for a mobile-origin ``complete``/``retired``/``error`` that the phone
    would be told about: the row still lands unread in the sidebar, inbox and
    badge. A needs-you row is never suppressed on the desktop for any origin
    (footnote 2: a duplicate is cheaper than a miss), and neither is anything
    the phone would not emit.
    """
    if not is_mobile_origin(state) or state.get("needs_user"):
        return False
    try:
        return phone_would_emit(state, config_dir)
    except Exception:  # noqa: BLE001 — a routing read must never cost a banner
        return False
