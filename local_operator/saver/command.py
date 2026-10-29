"""One conversation-scoped Radient Saver choice and its shared slash handler.

The global setting only makes Saver available. This module owns the tri-state
choice (``None`` follows the global/profile resolution) so the terminal, exec,
SDK and routed frontends all persist and report the same conversation state.
"""
from __future__ import annotations

from typing import Any

from local_operator.i18n.keys import wire_radient_saver as _rs
from local_operator.i18n.text import tr
from local_operator.saver.gate import Availability, resolve_saver

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


def availability() -> Availability | None:
    return Availability(True, True) if signed_in() else Availability(False, False)


def saver_on(session: Any) -> bool:
    settings = getattr(session, "routing_settings", None)
    if not isinstance(settings, dict):
        settings = getattr(session, "settings", None)
    decision = resolve_saver(
        settings,
        availability(),
        session_choice=getattr(session, "radient_saver_choice", None),
        agent_pin=getattr(session, "_saver_agent_pin", None),
    )
    return decision.on


def restore_saver_choice(session: Any) -> None:
    """Latest journal row wins; ``None`` is a real, durable inherit choice."""
    transcript = getattr(session, "_transcript", None)
    if transcript is None:
        return
    latest = transcript.latest_custom(SAVER_CUSTOM_TYPE)
    details = latest.get("details") if latest else None
    if isinstance(details, dict) and "enabled" in details:
        value = details["enabled"]
        if value is None or isinstance(value, bool):
            session.radient_saver_choice = value


async def set_saver_choice(session: Any, choice: bool | None) -> None:
    session.radient_saver_choice = choice
    await session._transcript.append_custom(SAVER_CUSTOM_TYPE, {"enabled": choice})


def saver_slash(session: Any, arg: str, SlashResult: Any) -> Any:
    """The single implementation behind ``/saver`` and ``/radient saver``."""
    if not signed_in():
        return SlashResult(kind="notice", text=_tr(_rs.notice_signed_out()), style="warning")
    choice = getattr(session, "radient_saver_choice", None)
    wanted = (arg or "").strip().lower()
    if wanted in {"status", "show"}:
        key = _rs.command_status_on if saver_on(session) else _rs.command_status_off
        return SlashResult(kind="notice", text=_tr(key()), style="info")
    if wanted in {"on", "enable", "enabled", "yes", "true"}:
        target: bool | None = True
    elif wanted in {"off", "disable", "disabled", "no", "false"}:
        target = False
    elif wanted in {"inherit", "global", "default", "clear"}:
        target = None
    elif not wanted:
        target = not saver_on(session)
    else:
        return SlashResult(kind="notice", text=_tr(_rs.command_usage()), style="warning")
    import asyncio

    async def persist() -> None:
        await set_saver_choice(session, target)

    try:
        asyncio.get_running_loop().create_task(persist())
    except RuntimeError:
        asyncio.run(persist())
    notify = getattr(session, "refresh_frontend_state", None)
    if callable(notify):
        notify()
    if target is None:
        return SlashResult(kind="notice", text=_tr(_rs.command_inherited()), style="info")
    # Report the account/global blockers at enable time rather than letting an
    # apparently successful switch silently do nothing.
    settings = getattr(session, "routing_settings", None) or getattr(session, "settings", None)
    decision = resolve_saver(settings, availability(), session_choice=target)
    if target and not decision.on:
        key = {
            "no_credit": _rs.notice_no_credit,
            "signed_out": _rs.notice_signed_out,
            "gate": _rs.notice_disabled,
            "auto_unavailable": _rs.notice_disabled,
        }.get(decision.source, _rs.notice_unavailable)
        return SlashResult(kind="notice", text=_tr(key()), style="warning")
    return SlashResult(
        kind="notice",
        text=_tr(_rs.command_enabled() if target else _rs.command_disabled()),
        style="info",
    )
