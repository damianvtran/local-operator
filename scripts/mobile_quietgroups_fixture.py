"""Serve synthetic quiet-group sessions, for capture.

Run:  PYTHONPATH=. .venv/bin/python scripts/mobile_quietgroups_fixture.py [PORT] <password>

Login at http://127.0.0.1:<port>. The password is NOT fixed and never printed:
pass it as the second argument (or set ``LOP_MOBILE_FIXTURE_PASSWORD``) — the
sibling fixtures' note on why a literal here would be a reusable credential.

Six sessions, one per state the relay web's quiet-group bar ships in slice S4
of the quiet-turn design (``docs/design/quiet-turns.md`` §5 + §8), tapped from
the list with the shot script beside this file:

* ``Quiet group collapsed``     — three peer receipts with a tool row between,
  the group AS THE TRANSCRIPT'S FIRST ROWS. That first-row position is
  load-bearing, not decoration: the bare count ``N+`` states a minimum only
  when the window's top edge cuts the group (``transcript.tsx``: ``partial =
  headCut && group.key === topGroup.key``, and ``topGroup`` is the group whose
  FIRST row is the visible window's first row). With anything visible above it
  the group reads a plain ``N``. First paint also carries ``hasMore = true``,
  so this session is the frame for ``Peer messages · 3+``.
* ``Quiet group resolved``      — the same shape over a scrollable tail: a real
  scroll-to-top gesture trips the transcript's auto-load, and this fixture
  answers history with an EMPTY page (no disk transcript is seeded — see
  below), so the settle flips ``hasMore`` false and the bar reads
  ``Peer messages · 2`` — the only way the bare count is honest on a top-cut
  group. The tail exists so the scroller can actually move: a fitting list
  fires no scroll event, and a gesture that cannot happen is not a capture.
* ``Quiet group expanded``      — the collapsed seed again; the shot script
  PRESSES the bar (the app's one disclosure idiom) and photographs the
  receipts with their tool rows, exactly as they paint at top level.
* ``Quiet group mixed``         — a peer receipt + a wake receipt adjacent:
  ``Messages · 2`` (the family word for more than one family, design §5's copy
  table).
* ``Quiet group wake only``     — two wake receipts: ``Wake messages · 2``.
* ``Quiet group single``        — ONE peer receipt: not a group at all, the
  ordinary card keeps its row. The control frame.

A SEVENTH STATE — the loaded-but-off-window window cut — was written and then
RETIRED by the first capture run on the folded tree (recorded on PR #2160):
since #2113 the transcript AUTO-GROWS its window the moment the top edge comes
into reach (``onScroll``: ``if (hiddenCount > 0) setWindowSize(n + PAGE)``),
and the client no longer renders the "show N more loaded" button at all — so
the ``N+``-over-loaded-rows state cannot be held still; the growth dissolves it
before any stable scroll position shows it. Its session was removed rather
than left photographing a state the shipped client cannot produce.

WHY NO ON-DISK TRANSCRIPTS ARE SEEDED. ``_history_page`` serves pages from
``config_dir()/sessions/<id>/transcript.jsonl``; absent, it answers
``([], False)``. That empty answer is what makes the resolved settle
deterministic: a seeded transcript would page real older rows in,
moving the very edge under test — and the durable fold does not classify peer
messages at all (``durable.py`` has no peer handling; the live projection
classifies them from ``details.sender``, ``projection.py``), so a disk seed
would resurrect them as plain user rows. The sessions are projection-only.

NO CONTROL OP IS PRESSED in any state (the bar is local disclosure state; the
scroll is paging), so unlike the imagegen fixture there is no request stub —
the real ``MobileDaemon`` serves as-is, with ``dial_registrants=False`` so this
never touches the operator's live daemon or their sessions. HOME and
LOCAL_OPERATOR_CONFIG_DIR are re-homed by ``scripts.probe_isolation`` on import.
"""

from __future__ import annotations

import asyncio
import os
import sys

import uvicorn

import scripts.probe_isolation  # noqa: F401  -- must be the first local import
from local_operator.mobile.daemon import MobileDaemon, SessionEntry, build_app
from local_operator.mobile.types import (
    EntryKind,
    SessionProjection,
    SessionRecord,
    TranscriptEntry,
)

#: The variable NAME (never a value) a caller may use instead of the second
#: argument — the same contract the sibling fixtures document.
FIXTURE_PASSWORD_ENV = "LOP_MOBILE_FIXTURE_PASSWORD"


def required_password(args: list[str]) -> str:
    """The password this run serves with, or a refusal naming the contract."""
    value = args[0] if args else os.environ.get(FIXTURE_PASSWORD_ENV, "")
    if not value:
        raise SystemExit(
            "this fixture needs a per-run password: pass it as the second argument, or "
            f"set {FIXTURE_PASSWORD_ENV}. Generate one with "
            "python -c 'import secrets;print(secrets.token_urlsafe(16))' and export it. "
            "It is never defaulted and never printed by this script."
        )
    return value


def _peer(
    entry_id: str, text: str, pid: int, name: str | None = None, model: str = "fixture"
) -> TranscriptEntry:
    """One inbound peer receipt. Sender fields are advisory by contract — a
    receipt with no name exercises the card's bare-label fallback."""
    sender: dict[str, object] = {"pid": pid, "model_label": model}
    if name:
        sender["conversation_name"] = name
    return TranscriptEntry(id=entry_id, kind="peer_message", text=text, details={"sender": sender})


def _wake(entry_id: str, text: str) -> TranscriptEntry:
    """One wake receipt — a ``notice`` carrying the wake marker the bar keys on."""
    return TranscriptEntry(id=entry_id, kind="notice", text=text, details={"notice_kind": "wake"})


def _tool(entry_id: str, call_id: str, summary: str) -> TranscriptEntry:
    """A finished, non-quiet tool row INSIDE a group — an action, not a split."""
    return TranscriptEntry(
        id=entry_id,
        kind="tool",
        tool_call_id=call_id,
        tool_name="bash",
        tool_state="done",
        summary=summary,
    )


def _line(entry_id: str, kind: EntryKind, text: str) -> TranscriptEntry:
    return TranscriptEntry(id=entry_id, kind=kind, text=text)


def _collapsed_entries() -> list[TranscriptEntry]:
    """The collapsed/expanded seed: the group opens the transcript (see the
    module doc for why that position is what produces ``N+``), one tool row
    inside so the expansion shows an action, three senders so the summary's
    top-2-plus-aggregate rule is exercised (the third has no name: the
    identity ladder's pid rung)."""
    return [
        _peer("qg-c-1", "Check-in done — nothing needs your attention today.", 880401, "Aida"),
        _tool("tc-qg-c-1", "qg-c-1", "session_report.sh --quiet"),
        _peer("qg-c-2", "Both sources are saved; the summary is in the vault.", 880402, "research"),
        _peer("qg-c-3", "The nightly batch finished clean.", 880403),
    ]


def _resolved_entries() -> list[TranscriptEntry]:
    """The resolved seed: the group at the top over a tail tall enough to
    scroll, so the shot's real drag fires the auto-load and the empty history
    answer settles ``hasMore`` false."""
    entries = [
        _peer("qg-r-1", "Overnight sweep landed — two flags, both resolved.", 880411, "Aida"),
        _peer("qg-r-2", "The digest is drafted; it needs nothing from you.", 880412, "research"),
    ]
    for index in range(10):
        entries.append(_line(f"qg-r-u{index}", "user", f"note {index + 1} — desk follow-up"))
        entries.append(_line(f"qg-r-a{index}", "assistant", "logged."))
    return entries


def _mixed_entries() -> list[TranscriptEntry]:
    """A peer receipt + a wake receipt adjacent — the mixed family."""
    return [
        _line("qg-m-1", "user", "Where did the overnight batch land?"),
        _peer("qg-m-2", "Finished clean; the report is pinned.", 880421, "Aida"),
        _wake("qg-m-3", "Wake fired: the morning sweep ran."),
    ]


def _wake_entries() -> list[TranscriptEntry]:
    return [
        _line("qg-w-1", "user", "Nudge me when the sweep lands."),
        _wake("qg-w-2", "Wake fired: morning sweep landed."),
        _wake("qg-w-3", "Wake fired: follow-up reminders queued."),
    ]


def _single_entries() -> list[TranscriptEntry]:
    """One receipt is not a group — the ordinary card is the control frame."""
    return [
        _line("qg-s-1", "user", "Any news?"),
        _peer("qg-s-2", "Nothing new — I'll flag it when something moves.", 880431, "Aida"),
    ]


def _projection(
    session_id: str, name: str, pid: int, entries: list[TranscriptEntry]
) -> SessionProjection:
    return SessionProjection(
        session_id=session_id,
        pid=pid,
        kind="tui",
        conversation_name=name,
        streaming=False,
        transcript=entries,
        version=2,
    )


async def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 4189
    password = required_password(sys.argv[2:])
    daemon = MobileDaemon(port=port, password=password, dial_registrants=False)
    for projection in (
        _projection("quietgroups-collapsed", "Quiet group collapsed", 900301, _collapsed_entries()),
        _projection("quietgroups-resolved", "Quiet group resolved", 900302, _resolved_entries()),
        _projection("quietgroups-expanded", "Quiet group expanded", 900303, _collapsed_entries()),
        _projection("quietgroups-mixed", "Quiet group mixed", 900304, _mixed_entries()),
        _projection("quietgroups-wake", "Quiet group wake only", 900305, _wake_entries()),
        _projection("quietgroups-single", "Quiet group single", 900306, _single_entries()),
    ):
        record = SessionRecord(
            pid=projection.pid,
            kind="tui",
            session_id=projection.session_id,
            conversation_name=projection.conversation_name,
            cwd="/synthetic",
            model_label="fixture",
            control_port=1,
            control_key="fixture",
        )
        entry = SessionEntry(record)
        entry.projection = projection
        _ensure_session_dir(projection.session_id)
        daemon.session_projections[projection.session_id] = projection
        daemon.table.entries[record.pid] = entry
    app = build_app(daemon)
    print(f"Fixture mobile: http://127.0.0.1:{port}", flush=True)
    await uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    ).serve()


def _ensure_session_dir(session_id: str) -> None:
    """Create the conversation directory the session LISTING requires.

    The same reason the imagegen fixture documents: ``daemon`` lists a live
    in-memory session only when ``config_dir()/sessions/<id>`` exists, so
    without this the list hides the sessions and every tap photographs the
    bare list instead of a bar.
    """
    from local_operator.paths import config_dir

    (config_dir() / "sessions" / session_id).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
