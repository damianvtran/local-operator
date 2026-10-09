"""Aida's first-run greeting: a four-state ledger and a HIDDEN trigger.

The desktop contract (design §4) has a ``greet`` op that must deliver ONE
onboarding turn and be idempotent; the TUI's first-run routing uses the same
function. This module owns the ledger that makes "once, and only to a person"
true, the trigger line the greeting turn starts from, and the integration-nudge
and tip ledgers the daily cadence reads.

THE STATE MACHINE (``onboarding.json`` ``greeting.state``)::

    owed ──request()──▶ requested ──arm lands──▶ armed ──fire──▶ delivered
      │                    ▲                        │
      │                    └──── pause cancels ─────┘
      └── skip (an install that already has human conversations) ──▶ skipped

* ``owed`` is the absent/initial state. NOTHING arms from it. The previous
  shape armed whenever ``greeted_at`` was ``None`` — from ``proactive.reconcile``
  and ``resume`` — so a headless runtime (a wake supervisor fire, ``lop exec``,
  the mobile daemon, ``lop serve`` with no window) could greet an install no
  person was looking at, and the first visible thing a user met was a turn that
  had already happened somewhere else (audit A1/A9).
* ``requested`` is written ONLY by an attended surface: the TUI's first-run
  routing / first contact and the desktop ``greet`` route. That is the one
  place "a person is in front of this right now" is known, so it is the one
  place allowed to start the greeting. ``reconcile``/``resume`` arm the row
  only from this state.
* ``armed`` means the one-shot ``aida-greeting`` row exists. A pause that
  cancels it moves it back to ``requested`` (the user did ask; they will be
  greeted on resume), never to ``owed``.
* ``delivered`` is stamped at the ACTUAL fire (``Session._deliver_wake`` calls
  :func:`mark_delivered`), not at arm time — so a row armed and then lost to a
  crash is not reported as a greeting the user saw. The daily cadence does not
  arm until this state (see :func:`cadence_allowed`): no 08:30 check-in from
  an assistant who has not yet met the user.
* ``skipped`` is terminal for an install that already had conversations when
  the ledger was first consulted: it is never greeted (R22) and its cadence
  behaves exactly as before this ledger existed.

MIGRATION. The pre-ledger file carried a bare ``greeted_at`` stamp written at
ARM time. An install with that stamp keeps working: it reads as ``delivered``
(:func:`greeting_state`), so its cadence is not withheld and it is never
greeted twice. The stamp is still written beside the state on delivery, so a
client reading only ``greeted_at`` keeps its meaning ("the greeting is done").

WHY THE GREETING RIDES THE WAKE PATH, AND WHY IT IS HIDDEN. There is no
turn-injection API on the messages route (it admits user turns only), and a
one-shot wake due now is the existing "start a turn in this conversation"
primitive that works with no live runtime. The row is armed ``hidden=True``:
the delivery writes ``details.hidden`` on the ``wake_prompt`` custom message,
which every human surface already skips (TUI live + replay, the desktop
history window, the mobile fold — ``harness.rows.is_hidden_wake_delivery``),
while the text still reaches the model. Her reply is the first visible row of
the conversation (audit A3/A4/U5). The trigger itself is a minimal FACTUAL line
(:func:`greeting_message`); the playbook lives in her packaged seed
(``agent_seeds/aida.md`` "First contact"), i.e. her stable instructions, never
in the conversation.

NO-PROVIDER REFUSAL. ``greet`` refuses when no provider is configured and
moves nothing, so the greeting fires after the user completes setup. The
predicate is the boot path's own hosting resolution, asked in a try/except so
unknown failure degrades to "not yet".
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from local_operator.aida import naming, state

logger = logging.getLogger(__name__)

#: Stable id for the greeting row, so a double-call race lands as a conflict
#: (treated as already-armed) rather than as two greetings.
GREETING_WAKE_ID = "aida-greeting"

#: The ledger states, in one place (see the module docstring's diagram).
GREETING_OWED = "owed"
GREETING_REQUESTED = "requested"
GREETING_ARMED = "armed"
GREETING_DELIVERED = "delivered"
GREETING_SKIPPED = "skipped"
GREETING_STATES = frozenset(
    {GREETING_OWED, GREETING_REQUESTED, GREETING_ARMED, GREETING_DELIVERED, GREETING_SKIPPED}
)

#: The attended surfaces allowed to request the greeting. Carried into the
#: trigger line so her first message can fit the surface (a terminal user is
#: told about `/aida`, a desktop user about the sidebar) without the engine
#: guessing.
SURFACE_TUI = "tui"
SURFACE_DESKTOP = "desktop"

#: How rarely she may nudge about setting up an integration (``aida.onboarding.
#: nudge_days``). The registry row's help text already points here; the engine
#: reads it through :func:`nudge_offer`.
DEFAULT_NUDGE_DAYS = 14

#: The permission clause the engine appends to the cadence message when the
#: nudge window is open (R25). The OFFER is the ledger event
#: (:func:`nudge_offer` stamps it), and her instructions admit a nudge only
#: when this sentence is present — so the bound is engine-enforced rather than
#: honour-system, and the ledger keeps its single writer.
NUDGE_CLAUSE = (
    "Integration nudge window: open. If ONE missing integration (an MCP server "
    "such as Google Workspace, Linear or Slack, or a similar set-up step) would be "
    "concretely useful to something in flight, you may suggest that single "
    "integration in this check-in and offer to set it up. Never more than one, and "
    "if nothing in flight needs one, do not bring integrations up."
)


def _ledger(config_dir: Path | str) -> dict[str, Any]:
    return state.read_json(state.onboarding_path(config_dir), what="onboarding") or {}


def _write_ledger(config_dir: Path | str, data: dict[str, Any]) -> None:
    state.write_json(state.onboarding_path(config_dir), data)


def greeting_state(config_dir: Path | str) -> str:
    """The greeting's ledger state, with the legacy stamp migrated on READ.

    A file written before the state machine carries only ``greeted_at`` (an
    ARM-time stamp): such an install already had its greeting armed under the
    old rules, so it reads ``delivered`` — never greeted twice, cadence not
    withheld. An unreadable or unknown value reads ``owed``, the state from
    which nothing arms, so a corrupt file can never start a greeting.
    """
    data = _ledger(config_dir)
    greeting = data.get("greeting")
    if isinstance(greeting, dict):
        value = greeting.get("state")
        if isinstance(value, str) and value in GREETING_STATES:
            return value
    if isinstance(data.get("greeted_at"), int) and not isinstance(data.get("greeted_at"), bool):
        return GREETING_DELIVERED
    return GREETING_OWED


def greeting_record(config_dir: Path | str) -> dict[str, Any]:
    """``{state, surface, requested_at, armed_at, delivered_at}`` for the routes.

    The desktop's ``GET /v1/desktop/aida`` exposes this as ``greeting`` so the
    renderer can tell "she is about to say hello" (requested/armed) from "she
    already has" (delivered) without inferring it from a boolean.
    """
    data = _ledger(config_dir)
    raw = data.get("greeting") if isinstance(data.get("greeting"), dict) else {}
    assert isinstance(raw, dict)

    def _int(key: str) -> int | None:
        value = raw.get(key)
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else None

    delivered_at = _int("delivered_at")
    legacy = data.get("greeted_at")
    if delivered_at is None and isinstance(legacy, int) and not isinstance(legacy, bool):
        delivered_at = legacy if greeting_state(config_dir) == GREETING_DELIVERED else None
    surface = raw.get("surface")
    return {
        "state": greeting_state(config_dir),
        "surface": surface if isinstance(surface, str) and surface else None,
        "requested_at": _int("requested_at"),
        "armed_at": _int("armed_at"),
        "delivered_at": delivered_at,
    }


def _set_greeting(config_dir: Path | str, new_state: str, now_ms: int, **fields: Any) -> None:
    """Move the ledger under the aida lock (read-modify-write of a shared file).

    The nudge and tip ledgers write the same file from the cadence path, so an
    unlocked write here could lose one of their stamps (or they one of ours).
    """
    with state.locked(config_dir):
        data = _ledger(config_dir)
        greeting = (
            dict(data.get("greeting") or {}) if isinstance(data.get("greeting"), dict) else {}
        )
        greeting["state"] = new_state
        stamp_key = {
            GREETING_REQUESTED: "requested_at",
            GREETING_ARMED: "armed_at",
            GREETING_DELIVERED: "delivered_at",
            GREETING_SKIPPED: "skipped_at",
        }.get(new_state)
        if stamp_key:
            greeting[stamp_key] = now_ms
        greeting.update({k: v for k, v in fields.items() if v is not None})
        data["greeting"] = greeting
        if new_state == GREETING_DELIVERED:
            # Kept for readers that predate the state (``greeted`` on the
            # desktop route, an older build sharing the config root): it now
            # means "delivered", which is what those readers always assumed.
            data["greeted_at"] = now_ms
        elif new_state in (GREETING_OWED, GREETING_REQUESTED, GREETING_ARMED):
            # A legacy stamp would otherwise make ``greeting_state`` of an
            # older reader claim delivery for a greeting still in flight.
            data.pop("greeted_at", None)
        _write_ledger(config_dir, data)


def greeting_message(config_dir: Path | str, *, surface: str = "") -> str:
    """The HIDDEN trigger for the greeting turn: facts only, no playbook.

    The line the model reads to know this turn is first contact. It is never
    shown to a person (the row is hidden on every surface), and it carries no
    instructions: those are her seed's "First contact" section, which is part
    of her stable prefix. Keeping prose out of the conversation is what lets a
    rename, a seed update or a later turn never contradict an instruction the
    transcript froze on day one (audit A7).

    ``signed_in_with=radient`` + ``identity=<name> <email>`` is what tells her
    NOT to ask for an email the Radient sign-in already recorded (audit A6);
    with no Radient identity she asks for it herself.
    """
    surface_word = surface or _ledger_surface(config_dir) or "unknown"
    parts = [f"surface={surface_word}"]
    identity = radient_identity(config_dir)
    if identity:
        parts.append("signed_in_with=radient")
        label = " ".join(
            bit
            for bit in (
                identity.get("name") or "",
                f"<{identity['email']}>" if identity.get("email") else "",
            )
            if bit
        )
        if label:
            parts.append(f"identity={label}")
    else:
        parts.append("signed_in_with=none")
    parts.append(f"assistant_name={naming.display_name(config_dir)}")
    return "[first-run] " + "; ".join(parts)


def _ledger_surface(config_dir: Path | str) -> str:
    raw = _ledger(config_dir).get("greeting")
    if isinstance(raw, dict) and isinstance(raw.get("surface"), str):
        return raw["surface"]
    return ""


def radient_identity(config_dir: Path | str) -> dict[str, str] | None:
    """``{"email", "name"}`` of the newest enabled Radient OAuth login, or ``None``.

    Read from the auth store's row for ``radient`` — the claims the login
    decoded out of the Radient ``id_token`` (``providers/oauth/radient.py``).
    Only an OAuth row counts: a pasted Radient key proves no identity. Never
    raises; an unreadable store is "not signed in", which only costs her one
    question.
    """
    try:
        from contextlib import closing

        from local_operator.providers.auth_store import AuthStore

        with closing(AuthStore(Path(config_dir) / "auth.db", config_dir=Path(config_dir))) as store:
            rows = [
                row for row in store.list_credentials("radient") if row.credential_type == "oauth"
            ]
    except Exception:  # noqa: BLE001 — identity is a courtesy, never a dependency
        logger.debug("aida: could not read the Radient identity", exc_info=True)
        return None
    for row in reversed(rows):
        email = str(row.data.get("email") or "").strip()
        name = str(row.data.get("name") or "").strip()
        if email or name:
            return {"email": email, "name": name}
    return None


def greeted_at(config_dir: Path | str) -> int | None:
    """When the greeting was DELIVERED, else ``None`` (legacy stamps included).

    Kept as the boolean-ish reader the desktop payload's ``greeted`` and older
    callers use. Under the state machine it answers only for ``delivered`` —
    a requested or armed greeting is not one the user has seen.
    """
    if greeting_state(config_dir) != GREETING_DELIVERED:
        return None
    return greeting_record(config_dir).get("delivered_at") or 0


def greeting_settled(config_dir: Path | str) -> bool:
    """Delivered or skipped: nothing more will ever happen to the greeting."""
    return greeting_state(config_dir) in (GREETING_DELIVERED, GREETING_SKIPPED)


def request_greeting(config_dir: Path | str, surface: str, *, now_ms: int | None = None) -> bool:
    """owed → requested, from an ATTENDED surface. Returns whether it moved.

    The ONLY entry into the greeting. Callers are the TUI (first-run routing,
    first contact with her conversation) and the desktop ``greet`` route —
    surfaces where a person is present. ``first_run_pending`` is the
    precondition: an install with human conversations is marked ``skipped``
    instead, so it can never be greeted later by any path.
    """
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    current = greeting_state(config_dir)
    if current != GREETING_OWED:
        return False
    try:
        if not fresh_install(config_dir):
            others = other_user_sessions(config_dir)
            if others:
                # R22 made durable: an existing install is never greeted, and
                # its cadence must not wait on a greeting that will never come.
                _set_greeting(config_dir, GREETING_SKIPPED, now)
            return False
        _set_greeting(config_dir, GREETING_REQUESTED, now, surface=surface)
        return True
    except Exception:  # noqa: BLE001 — a contended lock leaves it owed; retried next time
        logger.warning("aida: could not record the greeting request", exc_info=True)
        return False


def mark_greeted(config_dir: Path | str, now_ms: int) -> None:
    """requested → armed: the one-shot row now exists (the arm landed).

    Public because the engine's live-owner ensure arms the same row from
    :func:`proactive.reconcile` and owes the ledger the same fact: without it a
    later reconcile would find no row and arm a SECOND greeting. The name is
    kept for the callers it already had; it no longer means "delivered".
    """
    if greeting_state(config_dir) in (GREETING_DELIVERED, GREETING_SKIPPED):
        return
    _set_greeting(config_dir, GREETING_ARMED, now_ms)


def mark_delivered(config_dir: Path | str, now_ms: int | None = None) -> None:
    """armed → delivered, at the actual fire. Best-effort, never raises.

    Called by the session's wake delivery for the ``aida-greeting`` row, the one
    moment the greeting truly reaches her turn. Idempotent: a catch-up or a
    second fire after delivery moves nothing.
    """
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    try:
        if greeting_state(config_dir) == GREETING_DELIVERED:
            return
        _set_greeting(config_dir, GREETING_DELIVERED, now)
    except Exception:  # noqa: BLE001 — the turn runs regardless of the ledger
        logger.warning("aida: could not stamp the greeting delivery", exc_info=True)


def clear_greeted(config_dir: Path | str) -> None:
    """armed → requested: the row will not land, so the request stands instead.

    Two callers, and the second one is why this sentence is not about pauses
    alone (review round 2, N1): a PAUSE cancels the row before it fires, and
    the fire-time withhold (``Session._deliver_wake``) puts it back AFTER a
    fire that reached a runtime with no attended surface. Both mean the same
    thing to the user — she still has not said hello — so both leave the
    request standing rather than spending it.

    The user DID ask (an attended surface requested it), so the greeting stays
    theirs: the next resume — or the live owner's reconcile — arms it again.
    Moving it back to ``owed`` would make a pause during setup silently cost
    the greeting for good. Best-effort.
    """
    try:
        if greeting_state(config_dir) != GREETING_ARMED:
            return
        _set_greeting(config_dir, GREETING_REQUESTED, int(time.time() * 1000))
    except Exception:  # noqa: BLE001
        logger.warning("aida: could not re-owe the greeting", exc_info=True)


def greeting_armable(config_dir: Path | str) -> bool:
    """Whether an engine path may arm the row now: requested, and a provider.

    The ONE gate ``proactive.reconcile`` and ``resume`` ask. ``owed`` never
    arms — that is the whole headless-runtime fix.
    """
    return greeting_state(config_dir) == GREETING_REQUESTED and provider_configured(config_dir)


def cadence_allowed(config_dir: Path | str) -> bool:
    """Whether the daily cadence may arm: she has met the user, or never will.

    ``delivered``/``skipped`` allow it, and so does a legacy install (read as
    delivered). ``owed`` on an install with human conversations is migrated to
    ``skipped`` here, so an EXISTING install whose ledger predates this change
    keeps its check-in without waiting for a greeting that will never come.
    Fail-OPEN on error for that same population: a read failure must not
    silently stop an existing user's cadence.
    """
    try:
        current = greeting_state(config_dir)
        if current in (GREETING_DELIVERED, GREETING_SKIPPED):
            return True
        if current == GREETING_OWED:
            others = other_user_sessions(config_dir)
            if others or _her_conversation_had(config_dir):
                _set_greeting(config_dir, GREETING_SKIPPED, int(time.time() * 1000))
                return True
        return False
    except Exception:  # noqa: BLE001
        logger.warning("aida: cadence gate could not read the greeting ledger", exc_info=True)
        return True


def _sessions_root(config_dir: Path | str) -> Path:
    return Path(config_dir) / "sessions"


def _counts_as_operator_conversation(entry: Path, root: Path) -> bool:
    """Whether a user-classified session directory is a conversation HAD.

    Existence alone cannot answer it: the TUI boot materialises a session
    directory for every normal launch — the viewer's own id, holding only
    ``.execution-lease`` + ``.session.pid``, no transcript — so on a fresh
    install's FIRST boot the scan saw "a user session" and the predicate read
    false, and a provider-present first contact never greeted (UX review
    round 2, U4; the setup-exit path was unaffected because a setup-state boot
    creates no conversation directory at all).

    The discriminator is the engagement signal the runtime already answers
    this question with (``session_has_durable_history``: a real ``Message``
    row, never a custom one — wake prompts and quiet-dial peer notes persist
    without a turn), so "never engaged" reads as "no conversation had".
    Deliberately fail-closed around it, in R22's direction ("existing users
    are never re-routed"): a transcript with ANY bytes the reader will not
    call history — torn rows, a custom-only journal, a file it cannot open —
    counts as the operator's. Only the two shapes that positively say nothing
    was ever said here are discounted: no transcript at all, and an empty one.
    """
    from local_operator.session.runtime.engagement import (
        TRANSCRIPT_FILENAME,
        session_has_durable_history,
    )

    if session_has_durable_history(entry.name, root=root):
        return True
    try:
        return (entry / TRANSCRIPT_FILENAME).stat().st_size > 0
    except FileNotFoundError:
        return False
    except OSError:
        return True


def other_user_sessions(config_dir: Path | str) -> list[str] | None:
    """Ids of the operator's conversations besides hers, or ``None`` when unknown.

    The scan behind the first-run predicate (design §2.9): "no user sessions
    exist" means no ``is_user_session`` directory under ``sessions/`` other
    than the one ``aida/state.json`` names. Non-user origins (subagent runs,
    agent-shell sessions) deliberately do not count — they are machines'
    conversations, and an install whose store holds only those is still a
    fresh one as far as a human's first-run experience goes.

    Existence alone is not a conversation either: a directory counts only
    once it shows engagement (:func:`_counts_as_operator_conversation`) — the
    boot materialises one for every launch, and counting that made a fresh
    install look used the moment it started once (UX review round 2, U4).

    FAIL-CLOSED, in both ways a read can fail: an unreadable store answers
    ``None`` and an entry that cannot be classified counts as the operator's.
    The cost of the conservative direction is a first-run experience shown a
    moment later or not at all; the cost of the other is re-routing somebody
    with conversations (R22's "existing users are never re-routed"), which is
    the one outcome this predicate exists to prevent.
    """
    from local_operator.resume import is_user_session  # lazy: resume is heavy

    root = Path(config_dir)
    hers = state.session_id_of(root)
    try:
        entries = sorted(_sessions_root(root).iterdir())
    except FileNotFoundError:
        return []
    except OSError:
        logger.warning("aida: could not list the session store", exc_info=True)
        return None
    found: list[str] = []
    for entry in entries:
        if not entry.is_dir() or (hers and entry.name == hers):
            continue
        try:
            is_users = is_user_session(entry)
        except Exception:  # noqa: BLE001 — unclassifiable counts as the user's
            logger.warning("aida: could not classify session %s", entry.name, exc_info=True)
            is_users = True
        if is_users and _counts_as_operator_conversation(entry, root):
            found.append(entry.name)
    return found


def fresh_install(config_dir: Path | str) -> bool:
    """The first-run predicate: no human conversations ever, provider resolves.

    Both halves are load-bearing and both come from the design (§2.9): the
    sessions half keeps existing installs from being re-routed, and the
    provider half keeps the greeting from arming into a turn that cannot run
    (the same predicate :func:`greet` refuses on). Never raises: an unreadable
    store or config answers False, because "not fresh" is the direction that
    cannot re-route somebody.
    """
    try:
        others = other_user_sessions(config_dir)
    except Exception:  # noqa: BLE001 — a predicate, never a boot dependency
        logger.warning("aida: could not scan for user sessions", exc_info=True)
        return False
    if others is None or others:
        return False
    return provider_configured(config_dir)


def first_run_pending(config_dir: Path | str) -> bool:
    """Whether the first-run experience is still OWED to this install.

    ``fresh_install`` and the greeting ledger together, which is the gate the
    desktop contract states (design §4: "greeted=false and no user
    conversations other than hers"). "Not settled" rather than "owed": a user
    who was routed to her and quit before the greeting fired is routed to her
    again on the next attended boot, and the ledger (delivered/skipped) is
    what makes "once per config root" true even if the store is wiped later.
    """
    try:
        return fresh_install(config_dir) and not greeting_settled(config_dir)
    except Exception:  # noqa: BLE001 — same posture as :func:`fresh_install`
        logger.warning("aida: first-run predicate failed", exc_info=True)
        return False


def _her_conversation_had(config_dir: Path | str) -> bool:
    """Whether her OWN conversation already holds a real exchange.

    An install that met her before this ledger existed (opened her from
    ``/aida``, talked, never greeted) has nothing to be greeted for; the same
    engagement signal the sessions scan uses decides it.
    """
    hers = state.session_id_of(config_dir)
    if not hers or not (_sessions_root(config_dir) / hers).is_dir():
        return False
    # The POSITIVE signal only, unlike the other-sessions scan: her transcript
    # always holds bytes from birth (the bootstrap's birth and title custom
    # entries), so the scan's "any bytes count" fail-closed rule would mark
    # every fresh install as already-met and greet nobody. A real Message row
    # is what "they talked" means here.
    from local_operator.session.runtime.engagement import session_has_durable_history

    try:
        return session_has_durable_history(hers, root=_sessions_root(config_dir))
    except Exception:  # noqa: BLE001 — unknown: do not greet over a conversation
        return True


def provider_configured(config_dir: Path | str) -> bool:
    """Whether a provider/model pair is resolvable — the boot path's predicate.

    Deliberately the SAME resolution ``create_session`` performs
    (``resolve_hosting_model_with_source`` with no agent and no explicit
    selection), because "the greeting can run" must mean exactly "a session
    could boot here". Import is lazy: ``session_factory`` is heavy and this
    module is imported by boot paths. Unknown failures answer False: the
    refusal is retryable (nothing is stamped), so the safe direction of an
    unreadable config is "greet later", not "greet into a turn that cannot run".
    """
    import argparse

    try:
        from local_operator.config import ConfigManager
        from local_operator.session_factory import resolve_hosting_model_with_source

        args = argparse.Namespace(hosting=None, model=None, agent_name=None, agent_id=None)
        resolve_hosting_model_with_source(None, args, ConfigManager(config_dir=Path(config_dir)))
        return True
    except Exception:  # noqa: BLE001 — unconfigured or unreadable => not yet
        return False


async def greet(
    config_dir: Path | str,
    session_id: str,
    *,
    now_ms: int | None = None,
    surface: str = "",
) -> str:
    """Request (when attended) and arm the one-time greeting. Returns one word.

    ``surface`` names the ATTENDED caller (:data:`SURFACE_TUI` /
    :data:`SURFACE_DESKTOP`); only a call that names one can move the ledger out
    of ``owed``. The engine's own callers (``proactive.resume``) pass none and
    so can only arm a greeting a person already requested — never start one.

    Words: ``"greeted"`` (armed now), ``"already"`` (delivered, or armed by a racing
    second call), ``"skipped"`` (an install that already has conversations — she
    will never introduce herself here), ``"disabled"``, ``"paused"`` (requested
    and held; a resume arms it), ``"no-provider"``, ``"not-requested"`` (an
    unattended call on an owed greeting — the headless-runtime refusal),
    ``"owner"`` (a live runtime holds the session; its reconcile arms the
    requested row) or ``"failed"``.

    ``"skipped"`` is separate from ``"already"`` on purpose (review round 1,
    R-5): a route that renders both as "she has already introduced herself"
    tells the user a sentence that is false — on a skipped install she never
    will, and that is the correct behaviour, not a pending one.
    """
    from local_operator.aida import proactive

    root = Path(config_dir)
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    try:
        if state.env_disabled():
            return "disabled"
        pol = proactive.policy(root)
        if not pol.enabled:
            return "disabled"
        current = greeting_state(root)
        if current == GREETING_SKIPPED:
            return "skipped"
        if current == GREETING_DELIVERED:
            return "already"
        if current == GREETING_OWED:
            if not surface:
                return "not-requested"
            if not provider_configured(root):
                return "no-provider"
            if _her_conversation_had(root):
                _set_greeting(root, GREETING_SKIPPED, now)
                return "skipped"
            if not request_greeting(root, surface, now_ms=now):
                return "skipped" if greeting_state(root) == GREETING_SKIPPED else "failed"
            current = GREETING_REQUESTED
        if pol.paused:
            # Requested and HELD: the user asked, so the resume arms it.
            return "paused"
        if current == GREETING_ARMED:
            return "already"
        if not provider_configured(root):
            return "no-provider"

        from local_operator.wakes.arm import WakeWriteError, arm_wake

        try:
            await arm_wake(
                root,
                session_id,
                {"message": greeting_message(root, surface=surface), "in": "1s"},
                wake_id=GREETING_WAKE_ID,
                now_ms=now,
                hidden=True,
            )
        except WakeWriteError as exc:
            # 409 = the row already exists (a racing second greet): the
            # greeting IS armed, which is the fact this call exists to
            # establish. 503 = a live owner; the state stays ``requested`` so
            # the owner's own reconcile arms it (``greeting_armable``).
            if exc.status == 409:
                mark_greeted(root, now)
                return "greeted"
            if exc.status == 503:
                return "owner"
            logger.warning("aida: greeting arm refused: %s", exc)
            return "failed"
        mark_greeted(root, now)
        return "greeted"
    except Exception:  # noqa: BLE001 — a greeting must never fail its caller
        logger.warning("aida: greet failed", exc_info=True)
        return "failed"


# --------------------------------------------------------------------------- #
# The integration-nudge ledger (R25) — the engine's half, read open/closed
# --------------------------------------------------------------------------- #


def _nudge_days(config_dir: Path | str) -> int:
    """``aida.onboarding.nudge_days``, best-effort, floored at one day.

    Read through the settings facade (the registry row's own reader), so the
    CLI, ``/settings`` and this bound cannot disagree about the value; an
    unreadable config answers the default rather than costing the bound, and
    a value below the registry's minimum answers it too (a hand-edited 0 must
    not turn the nudge into every-day spam).
    """
    try:
        from local_operator import settings_io
        from local_operator.config import ConfigManager

        setting = settings_io.BY_KEY.get("aida.onboarding.nudge_days")
        if setting is None:  # pragma: no cover - registry guarantee
            return DEFAULT_NUDGE_DAYS
        days = int(settings_io.read_setting(ConfigManager(config_dir=Path(config_dir)), setting))
    except Exception:  # noqa: BLE001 — the default IS the bound of last resort
        logger.debug("aida: could not read nudge_days", exc_info=True)
        return DEFAULT_NUDGE_DAYS
    return days if days >= 1 else DEFAULT_NUDGE_DAYS


def nudge_offer(config_dir: Path | str, *, now_ms: int | None = None) -> str | None:
    """The nudge clause for a cadence row being ARMED now, or ``None`` if not due.

    One function, two effects, and they are deliberately inseparable: when the
    window is open it stamps the ledger (``nudge_offered_at``,
    ``nudge_offers``) and returns :data:`NUDGE_CLAUSE` for the row's message;
    when it is closed — or the stamp cannot be recorded — it returns ``None``.
    That ordering is what makes the bound real rather than advisory: the
    permission and its ledger move together, and a clause can never exist
    without the stamp that spends the window.

    WHO SENDS THE NUDGE. The offer rides the CADENCE ROW (its message is
    persisted in the transcript), so the moment of truth is the fire, not this
    call — and the engine, which cannot read her prose, records the offer
    rather than an actual suggestion. An offer that never turns into a nudge
    is therefore spent conservatively (the next window opens in
    ``nudge_days``); the alternative — an honour-system ledger she writes —
    loses a racing read-modify-write of ``onboarding.json`` against the
    greeting's own writers, and R25's "bounded" would stop being enforceable.

    The read-modify-write runs under the aida lock so two processes arming the
    cadence at once cannot both spend one window. Never raises: a contended
    lock answers ``None`` (no offer without a stamp — the safe direction).
    """
    root = Path(config_dir)
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    window_ms = _nudge_days(root) * 86_400_000
    try:
        with state.locked(root):
            path = state.onboarding_path(root)
            data = state.read_json(path, what="onboarding") or {}
            last = data.get("nudge_offered_at")
            last_ms: int | None = None
            if isinstance(last, (int, float)) and not isinstance(last, bool):
                last_ms = int(last)
            if last_ms is not None and now - last_ms < window_ms:
                return None
            data["nudge_offered_at"] = now
            data["nudge_offers"] = int(data.get("nudge_offers") or 0) + 1
            state.write_json(path, data)
    except Exception:  # noqa: BLE001 — a lock refusal must not cost the cadence
        logger.warning("aida: could not record the nudge window", exc_info=True)
        return None
    return NUDGE_CLAUSE


# --------------------------------------------------------------------------- #
# The daily tip ledger (audit A10/U15/D14) — one useful fact on a quiet day
# --------------------------------------------------------------------------- #

#: The shortest gap between two tips, in milliseconds. 20 hours rather than a
#: calendar day on purpose: the cadence fires once a day at a fixed local time,
#: and a check-in that slips by an hour (a laptop asleep at 08:30) would
#: otherwise skip a whole day's tip. The cost of the shorter window is bounded
#: by the pool draining rather than repeating — review round 1, Q2 caught this
#: documented as "one per calendar day" when the code said 20 hours.
TIP_WINDOW_MS = 20 * 3_600_000

#: Tip ids in the order they are offered, each with the fact that makes it
#: APPLICABLE (a tip the install has already acted on is skipped, never
#: offered) and the clause text. One clause per check-in, and a tip id is
#: offered at most once per install (``tips_given``), so the pool drains rather
#: than nags. The texts name commands this build ships.
TIP_MOBILE_RADIENT = "mobile-radient"
TIP_MOBILE_SIGNIN = "mobile-signin"
TIP_FIRST_TEAM = "first-team"
TIP_INTEGRATIONS = "integrations"
TIP_PAUSE = "pause"

_TIP_TEXT: dict[str, str] = {
    TIP_MOBILE_RADIENT: (
        "Phone access is not set up. They are signed in with Radient, so the "
        "easiest route is the Radient relay — a private sign-in-protected URL for "
        "their phone, essentially free at the current price (USD 0/month; quote it "
        "with `lop tunnel billing`). Offer to set it up (guide: mobile)."
    ),
    TIP_MOBILE_SIGNIN: (
        "Phone access is not set up and they are not signed in with Radient. The "
        "recommended, least technical route is `/login radient` and then the Radient "
        "relay; the alternative is self-hosting a Cloudflare tunnel in front of "
        "`lop mobile` (guide: tunnel). Offer whichever suits them."
    ),
    TIP_FIRST_TEAM: (
        "They have no team yet. A team owns one domain of work and is reused for "
        "every request in it; offer to set up their first one around what they "
        "work on."
    ),
    TIP_INTEGRATIONS: (
        "You can connect their tools — Google Workspace, Linear, Slack and other "
        "MCP integrations — so check-ins and delegated work can see them. Mention it "
        "once; offer to set one up."
    ),
    TIP_PAUSE: (
        "Remind them, once, that `/aida pause` stops these check-ins and "
        "`/aida resume` brings them back."
    ),
}

#: The clause wrapper, so her instructions can key on one stable prefix.
TIP_CLAUSE_PREFIX = "Tip for a quiet day (use only if nothing needs action): "


def _tip_applicable(config_dir: Path, tip_id: str) -> bool:
    """Whether a tip still describes something the install has NOT done.

    Fact-backed and fail-CLOSED: a predicate that cannot read its fact answers
    False (skip the tip), because a tip that tells someone to set up what they
    already have reads as an assistant that is not paying attention.
    """
    try:
        if tip_id in (TIP_MOBILE_RADIENT, TIP_MOBILE_SIGNIN):
            if (config_dir / "tunnel" / "config.json").exists():
                return False
            signed_in = radient_identity(config_dir) is not None or _radient_oauth_row(config_dir)
            return signed_in if tip_id == TIP_MOBILE_RADIENT else not signed_in
        if tip_id == TIP_FIRST_TEAM:
            teams = config_dir / "teams"
            return not (teams.is_dir() and any(teams.iterdir()))
        if tip_id == TIP_INTEGRATIONS:
            mcp = config_dir / "mcp.json"
            if not mcp.exists():
                return True
            import json as _json

            servers = _json.loads(mcp.read_text(encoding="utf-8")).get("mcpServers") or {}
            return not servers
        return tip_id == TIP_PAUSE
    except Exception:  # noqa: BLE001 — fail closed: skip the tip
        logger.debug("aida: tip predicate %s failed", tip_id, exc_info=True)
        return False


def _radient_oauth_row(config_dir: Path) -> bool:
    """A Radient OAuth login with no decoded identity still counts as signed in."""
    from contextlib import closing

    from local_operator.providers.auth_store import AuthStore

    with closing(AuthStore(config_dir / "auth.db", config_dir=config_dir)) as store:
        return any(row.credential_type == "oauth" for row in store.list_credentials("radient"))


def tip_offer(config_dir: Path | str, *, now_ms: int | None = None) -> str | None:
    """The tip clause for a cadence row being ARMED now, or ``None``.

    The sibling of :func:`nudge_offer`, with the same two-effects-inseparable
    rule: the first applicable, never-given tip is stamped into the ledger
    (``tips_given`` + ``tip_offered_at``) in the same locked write that hands
    back its clause, so a clause never exists without the stamp that spends
    it. At most one tip per :data:`TIP_WINDOW_MS` (a reconcile can rebuild a
    cadence row more than once a day), and never before the greeting is
    delivered — the tips are for someone she has met. Never raises; a contended
    lock answers ``None``.
    """
    root = Path(config_dir)
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    try:
        if greeting_state(root) not in (GREETING_DELIVERED, GREETING_SKIPPED):
            return None
        with state.locked(root):
            path = state.onboarding_path(root)
            data = state.read_json(path, what="onboarding") or {}
            last = data.get("tip_offered_at")
            if isinstance(last, int) and not isinstance(last, bool) and now - last < TIP_WINDOW_MS:
                return None
            given = [str(t) for t in data.get("tips_given") or [] if isinstance(t, str)]
            choice = next(
                (t for t in _TIP_TEXT if t not in given and _tip_applicable(root, t)), None
            )
            if choice is None:
                return None
            # Both mobile tips describe ONE suggestion; once either is given
            # the other is spent too, or a later sign-in would repeat it.
            spent = [choice]
            if choice in (TIP_MOBILE_RADIENT, TIP_MOBILE_SIGNIN):
                spent = [TIP_MOBILE_RADIENT, TIP_MOBILE_SIGNIN]
            data["tips_given"] = given + [t for t in spent if t not in given]
            data["tip_offered_at"] = now
            state.write_json(path, data)
    except Exception:  # noqa: BLE001 — a ledger miss must not cost the cadence
        logger.warning("aida: could not record the tip", exc_info=True)
        return None
    return TIP_CLAUSE_PREFIX + _TIP_TEXT[choice]
