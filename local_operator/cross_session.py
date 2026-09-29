"""One reader for ``display.hide_cross_session``, shared by TUI and phone.

WHY A MODULE. The flag gates rows on two surfaces in two different packages —
the TUI's transcript builders (``tui/app.py``, ``tui/session_presentation.py``,
``tui/widgets/subagent_view.py``) and the phone daemon's projection fold
(``mobile/projection.py``). If each gate read the key itself, a rename or a
polarity fix would have to land in five places; one predicate means the
surfaces cannot disagree about what "hidden" means.

THE HIDDEN SET, exactly (the frozen cross-repo contract):
transcript custom rows whose ``custom_type == "peer_message"`` and tool rows
whose tool name is exactly ``"send"``. Nothing else — wake receipts, hub
messages and the receiver-side peer model-switch audit card stay visible, the
model still receives peer messages (this is a view filter for the human
reader, not a privacy boundary or a data-plane change), and raw-journal
surfaces (``jobs``, ``hub`` peek, the picker's verbose preview) keep the rows.
It is NOT an activity filter either: an attention signal derived from the
transcript's mere presence or mtime — the phone list's unseen dot is
mtime-based — still reads hidden traffic as activity. That is accepted rather
than a leak: no content, count or preview of a hidden row travels (QA round 1
on #1746, O2), and silencing it would make the flag a data-plane change.

The reader is the TUI's EXISTING process cache (``tui/settings.py``), whose
invalidators already exist: ``settings_reload`` runs after every write and the
config watcher calls it on another process's write — so this flag needs no
cache, no config path and no reload plumbing of its own.
"""

from __future__ import annotations

#: OFF is today's rendering: the flag is an opt-in HIDER, so an absent key
#: must paint every cross-session row. The registry (``settings_io``) states
#: the same default, and ``tests/unit/test_settings_io.py`` pins the two
#: constants together so a drift is a red test rather than a page that lies.
DEFAULT_HIDE_CROSS_SESSION = False

#: The hidden tool row's name, exactly as ``tools/registry.py`` registers it.
#: Gates compare against this constant rather than the literal so the rename
#: site is one line; the whole ``send`` tool is hidden (message and
#: model-switch modes alike) — one predicate, no argument sniffing.
SEND_TOOL_NAME = "send"


def cross_session_hidden() -> bool:
    """Whether cross-session rows are due to be hidden on this surface.

    The import is function-local and not incidental: this module is imported
    from the mobile daemon's import path and from the TUI's paint-heavy
    modules, and neither should pay for the display-flag reader at import
    time. Most callers read once per handler or pass; the phone fold's
    ``ProjectionFold._tool_row`` deliberately reads per minted call, which is
    fine because this is the process-cached reader — not a config parse — and
    the fold's rate is per decision, not per frame (review round 1, N2).
    """
    from local_operator.tui.settings import settings_get

    return bool(settings_get("display.hide_cross_session", DEFAULT_HIDE_CROSS_SESSION))
