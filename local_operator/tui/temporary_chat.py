"""The terminal's temporary-chat verbs: ``/new temporary``, ``/temporary``, ``lop --temporary``.

Pure functions over a config root and a session id, free of Textual, so the
app's handlers stay thin and the copy is tested without an app. The mark and
its rules live in :mod:`local_operator.session.temporary`; this module only
turns its answers into the terminal's keyed sentences.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from local_operator.i18n.keys import tui_app as _ta
from local_operator.i18n.keys import tui_notices as _tn
from local_operator.i18n.text import tr
from local_operator.session import temporary

#: How long ``/temporary on`` waits for its repeat on a chat that has messages.
CONFIRM_WINDOW_S = 30.0

_ON_WORDS = frozenset({"on", "yes", "true", "enable", "enabled"})
_OFF_WORDS = frozenset({"off", "no", "false", "disable", "disabled", "keep"})


def parse(arg: str) -> str | None:
    """``"on"``, ``"off"``, ``"status"``, ``"toggle"`` (bare), or ``None`` (unknown word)."""
    word = arg.strip().casefold()
    if not word:
        return "toggle"
    if word in ("status", "show"):
        return "status"
    if word in _ON_WORDS:
        return "on"
    if word in _OFF_WORDS:
        return "off"
    return None


def is_temporary(root: Path, session_id: str) -> bool:
    return bool(session_id) and temporary.is_marked(root, session_id)


def hold_text(code: str) -> str:
    """The keyed phrase for a hold code (``tui.app.temporary_hold_<code>``)."""
    accessor = getattr(_ta, f"temporary_hold_{code}", None)
    if accessor is None:
        accessor = _ta.temporary_hold_cannot_verify
    return str(tr("tui.app")(accessor()))


def deletes_in(info: dict[str, Any] | None, now: float | None = None) -> str:
    """``deletes in N min`` or the hold phrase, for a row's ``temporary`` object."""
    if not info:
        return ""
    if info.get("hold"):
        return hold_text(str(info["hold"]))
    deletes_at = info.get("deletes_at")
    if not isinstance(deletes_at, (int, float)):
        return ""
    clock = time.time() if now is None else now
    minutes = max(1, int((deletes_at - clock + 59) // 60))
    return str(tr("tui.app")(_ta.temporary_deletes_in(minutes=minutes)))


def status_sentence(root: Path, session_id: str, now: float | None = None) -> str:
    """The ``/temporary status`` answer."""
    _t = tr("tui.notices")
    info = temporary.RowProjector(root, now=now).info(session_id)
    if info is None:
        return str(_t(_tn.temporary_status_off()))
    return str(_t(_tn.temporary_status_on(deletes_in=deletes_in(info, now))))


def apply(root: Path, session_id: str, on: bool) -> tuple[bool, str]:
    """Mark or keep ``session_id``; ``(ok, sentence)`` in the operator's language.

    A brand-new session may not have written its directory yet, so marking
    creates it (the runtime materialises into the same directory).
    """
    _t = tr("tui.notices")
    if on:
        (Path(root) / "sessions" / session_id).mkdir(parents=True, exist_ok=True)
    result = temporary.write_mark(root, session_id, on, by="tui")
    if result.ok:
        if on:
            minutes = temporary.configured_idle_minutes(root)
            return True, str(_t(_tn.temporary_on(minutes=minutes)))
        return True, str(_t(_tn.temporary_off()))
    if result.code == temporary.CODE_PINNED:
        return False, str(_t(_tn.temporary_refused_pinned()))
    if result.code == temporary.CODE_IN_PROJECT:
        return False, str(_t(_tn.temporary_refused_project()))
    from local_operator.server.utils.desktop_sessions import TemporaryRefusal

    reason = TemporaryRefusal(result.code, session_id).envelope().get("text") or result.code
    return False, str(_t(_tn.temporary_refused(reason=reason)))


def pin_refusal_sentence(root: Path, session_id: str) -> str:
    """The sentence refusing a pin on a temporary chat, or ``""``."""
    if not temporary.pin_refusal(root, session_id):
        return ""
    from local_operator.server.utils.desktop_sessions import TemporaryRefusal

    return str(TemporaryRefusal(temporary.CODE_TEMPORARY, session_id).envelope().get("text"))


def deleted_sentence(root: Path, session_id: str) -> str:
    """The tombstone sentence for a deleted temporary chat, or ``""``."""
    if temporary.deleted_at(root, session_id) is None:
        return ""
    return str(tr("tui.notices")(_tn.temporary_deleted(session_id=session_id)))


def usage_hint() -> str:
    return str(tr("tui.notices")(_tn.temporary_new_hint()))
