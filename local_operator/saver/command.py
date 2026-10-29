"""One conversation-scoped Radient Saver choice and its shared slash handler.

The global setting only makes Saver available. This module owns the tri-state
choice (``None`` follows the global/profile resolution) so the terminal, exec,
SDK and routed frontends all persist and report the same conversation state.

ONE NAMING PER FACT (review round 1, M1/m1). Three attributed names once carried
the same three inputs — ``_saver_agent_pin`` (nothing wrote it) beside
``_saver_agent_pin_value`` (the live one), and ``_saver_session_choice`` (dead)
beside ``radient_saver_choice`` — so ``/saver status`` and the request path could
answer the same question differently, and did. :func:`saver_choices` is now the
only place those fields are named, and the turn path
(``model/configure.py::_saver_choices``) calls it rather than reading them.

TWO ANSWERS, DELIBERATELY (review M3 / UX U3). :func:`saver_state` reports the
session's own state for the badge and ``/saver status``; the turn path keeps
calling ``gate.resolve_saver`` with the account's balance, because only that
path decides whether a request splices. See ``gate.resolve_saver_setting``.

THE WRITE OBEYS THE GATE (review M4). ``set_saver_choice`` refuses a choice the
gate will not honour, so no caller — slash, ``--saver``, ``SessionSpec`` — can
journal an enable the copy just reported as refused, and there is no per-caller
"do not write" rule left to drift.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from local_operator.i18n.keys import wire_radient_saver as _rs
from local_operator.i18n.text import tr
from local_operator.saver.gate import (
    AVAILABILITY_DEADLINE_S,
    Availability,
    SaverDecision,
    read_availability,
    resolve_saver,
    resolve_saver_setting,
)

SAVER_CUSTOM_TYPE = "radient_saver_choice_v1"


def _tr(message: Any) -> str:
    return str(tr("wire.radient_saver")(message))


def signed_in() -> bool:
    """Use the credential gate's own sign-in test; never expose the credential."""
    from local_operator.tunnels.api import credential_id

    try:
        credential_id(None)
        return True
    except Exception:  # noqa: BLE001 — no exactly-one usable Radient login
        return False


def session_settings(session: Any) -> dict[str, Any] | None:
    """The mapping ``configured_mode`` reads, from either session shape."""
    settings = getattr(session, "routing_settings", None)
    if not isinstance(settings, dict):
        settings = getattr(session, "settings", None)
    return settings if isinstance(settings, dict) else None


def saver_choices(
    session: Any, *, stream: Any = None
) -> tuple[bool | None, bool | None, bool | None]:
    """``(session choice, agent pin, parent value)`` — the only naming of them.

    ``stream`` is the resolver's own stream when the caller is a fork: a
    ``SessionStreamFn`` binds another session's live state for the CHOICE but
    carries the parent's spawn value itself (``harness/subagent.py`` seeds
    ``_saver_parent_value`` on the stream, not the session).
    """
    if stream is None:
        stream = getattr(session, "_stream_fn", None)
    return (
        getattr(session, "radient_saver_choice", None),
        getattr(session, "_saver_agent_pin_value", None),
        getattr(stream, "_saver_parent_value", None),
    )


def cached_signed_in(session: Any) -> bool | None:
    """The last sign-in answer this session observed; ``None`` before any.

    NOT read live here on purpose. ``signed_in()`` reaches the credential store,
    which on macOS shells out to ``security`` — and this function's callers
    include the BAND's paint path, where an external process per repaint is both
    a stop-the-world cost and an external effect the rendered-frame tests
    deliberately trap (``test_host_speed_settings_rendered.py`` caught it as
    ``external-effect-or-provider``). The answer is stamped by
    ``refresh_availability`` at the moments an account read is legitimate —
    enable, ``/saver status``, ``--saver``, a resume's first turn, a 402.
    """
    value = getattr(session, "_saver_signed_in", None)
    return value if isinstance(value, bool) else None


def saver_state(session: Any) -> SaverDecision:
    """The session's own state — what the badge and ``/saver status`` report."""
    session_choice, agent_pin, parent_value = saver_choices(session)
    return resolve_saver_setting(
        session_settings(session),
        signed_in=cached_signed_in(session) is True,
        session_choice=session_choice,
        agent_pin=agent_pin,
        parent_value=parent_value,
    )


def observe_availability(session: Any, availability: Availability | None) -> None:
    """Stamp the two cached facts a read answers, if it answered at all.

    Tolerates a stream with no bound session: the turn path can resolve over a
    stream before ``Session`` binds it (test rigs and the pre-bind window), and
    a cache write must never be the thing that fails a turn.
    """
    if availability is None or session is None:
        return
    session._saver_availability = availability
    session._saver_signed_in = availability.signed_in


def saver_on(session: Any) -> bool:
    return saver_state(session).on


def cached_availability(session: Any) -> Availability | None:
    """The last balance read this session made, if any (never a guess)."""
    value = getattr(session, "_saver_availability", None)
    return value if isinstance(value, Availability) else None


async def read_gate_availability() -> Availability | None:
    """The gate's real account half under its own deadline; ``None`` unknown.

    Deliberately not wrapped in a wider try: ``read_availability`` already
    answers ``None`` for every transport failure. This only bounds the total
    time, which is the deadline the gate documents (:data:`AVAILABILITY_DEADLINE_S`)
    and the reason a user's enable can wait at all.
    """
    try:
        async with asyncio.timeout(AVAILABILITY_DEADLINE_S):
            return await read_availability()
    except (TimeoutError, asyncio.CancelledError):
        return None


async def refresh_availability(session: Any) -> Availability | None:
    """Read the account half and cache it on the session; ``None`` stays uncached."""
    availability = await read_gate_availability()
    observe_availability(session, availability)
    return availability


def restore_saver_choice(session: Any) -> None:
    """Latest journal row wins; ``None`` is a real, durable inherit choice.

    ``latest_custom`` already returns the row's ``details`` mapping — reading
    ``["details"]`` off it found nothing, so a resume silently dropped the
    choice while the row sat on disk (QA Q2).
    """
    transcript = getattr(session, "_transcript", None)
    if transcript is None:
        return
    details = transcript.latest_custom(SAVER_CUSTOM_TYPE)
    if not isinstance(details, dict) or "enabled" not in details:
        return
    value = details["enabled"]
    if value is None or isinstance(value, bool):
        session.radient_saver_choice = value


@dataclass(frozen=True)
class SaverWrite:
    """One attempted write: what is now in effect, and whether it was journaled."""

    written: bool
    #: The decision the gate reached for the write — the refusal's reason when
    #: ``written`` is False, and the state now in effect when it is True.
    decision: SaverDecision


async def set_saver_choice(
    session: Any,
    choice: bool | None,
    *,
    availability: Availability | None = None,
) -> SaverWrite:
    """Set and journal this session's choice — the ONE write path.

    A REFUSED ENABLE WRITES NOTHING (review M4). ``True`` is journaled only when
    the gate would honour it, which needs the account's balance: ``availability``
    is that read when the caller already made one at a user-initiated moment,
    and is made here otherwise. ``False`` and ``None`` are always writable — they
    only ever reduce what leaves the machine, so they need no account fact.

    The journal write is AWAITED BEFORE the in-memory assignment (review M2), so
    nothing can publish a state the transcript does not hold.
    """
    _previous, agent_pin, parent_value = saver_choices(session)
    if choice is True:
        if availability is None:
            availability = await refresh_availability(session)
        if availability is None:
            # ``resolve_saver`` would call this "signed_out", because an unknown
            # account half is refused at the same step (fail-closed). That is
            # right for the REQUEST and wrong for the COPY: a network blip must
            # not tell a signed-in user to log in.
            source = "signed_out" if not signed_in() else "unknown_balance"
            return SaverWrite(written=False, decision=SaverDecision(False, source))
        decision = resolve_saver(
            session_settings(session),
            availability,
            session_choice=True,
            agent_pin=agent_pin,
            parent_value=parent_value,
        )
        if not decision.on:
            return SaverWrite(written=False, decision=decision)
    await session._transcript.append_custom(SAVER_CUSTOM_TYPE, {"enabled": choice})
    session.radient_saver_choice = choice
    return SaverWrite(
        written=True,
        decision=resolve_saver_setting(
            session_settings(session),
            signed_in=signed_in(),
            session_choice=choice,
            agent_pin=agent_pin,
            parent_value=parent_value,
        ),
    )


#: ``SaverDecision.source`` -> the notice that explains it. One map, so a new
#: refusal source cannot silently fall back to "unavailable".
_REFUSAL_NOTICE = {
    "signed_out": _rs.notice_signed_out,
    "no_credit": _rs.notice_no_credit,
    "unknown_balance": _rs.notice_unknown_balance,
    "gate": _rs.notice_disabled,
    "auto_unavailable": _rs.notice_disabled,
}


def refusal_text(decision: SaverDecision) -> str:
    """The sentence for a refused write, from the same map the slash uses.

    Exec and the SDK have no transcript notice to put it in, so without this
    they would either swallow the refusal or invent a second wording for it.
    """
    key = _REFUSAL_NOTICE.get(decision.source, _rs.notice_unavailable)
    return _tr(key())


async def saver_slash(session: Any, arg: str, SlashResult: Any) -> Any:
    """The single implementation behind ``/saver`` and ``/radient saver``.

    ASYNC BECAUSE THE ANSWER DEPENDS ON THE WRITE (review M2). The old sync
    handler scheduled the journal write and answered in the same tick, so both
    frontends published the pre-write state next to a notice claiming the new
    one. Awaiting it is what orders assign -> journal -> publish.

    READING AND CLEARING WORK SIGNED OUT (UX U6). ``status`` and ``inherit``
    only report or reduce, so the sign-out guard would deny a user the one view
    that says what a stored choice is doing — and the one word that clears it.
    Turning Saver ON still requires the account, which is the gate's own step.
    """
    wanted = (arg or "").strip().lower()
    if wanted in {"status", "show"}:
        # A report is a user-initiated moment, so it takes the real read rather
        # than answering from a cache that may predate the balance running out.
        await refresh_availability(session)
        key = _rs.command_status_on if saver_state(session).on else _rs.command_status_off
        return SlashResult(kind="notice", text=_tr(key()), style="info")
    if wanted in {"inherit", "global", "default", "clear"}:
        await set_saver_choice(session, None)
        return SlashResult(kind="notice", text=_tr(_rs.command_inherited()), style="info")
    if wanted in {"on", "enable", "enabled", "yes", "true"}:
        target: bool | None = True
    elif wanted in {"off", "disable", "disabled", "no", "false"}:
        target = False
    elif not wanted:
        target = not saver_state(session).on
    else:
        return SlashResult(kind="notice", text=_tr(_rs.command_usage()), style="warning")
    if not signed_in():
        # Before any write and before any read: the sign-in test is local, and
        # the gate's own first step refuses on it, so this only turns the same
        # answer into the sentence that names the remedy.
        return SlashResult(kind="notice", text=_tr(_rs.notice_signed_out()), style="warning")
    # NO SECOND DECISION HERE (review M4). The writer reads the account half when
    # the enable needs it and refuses with the reason; duplicating that read in
    # this handler is how the "do not write" rule came to live in one caller.
    result = await set_saver_choice(session, target)
    if not result.written:
        return SlashResult(
            kind="notice", text=refusal_text(result.decision), style="warning"
        )
    notify = getattr(session, "refresh_frontend_state", None)
    if callable(notify):
        notify()
    return SlashResult(
        kind="notice",
        text=_tr(_rs.command_enabled() if target else _rs.command_disabled()),
        style="info",
    )
