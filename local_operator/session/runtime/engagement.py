"""Whether a session has been ENGAGED — has it run a real turn yet?

The evidence behind ``SessionRecord.started``. The live half of the answer is
the record bit itself (``registry`` reads it, ``RuntimeServer.set_record_started``
flips it, ``TuiSessionHandle.rebind`` re-seeds it); this module is the half that
has to answer for a session NOBODY IS RUNNING, where there is no record and no
bit — only the session's own transcript on disk.

WHY THIS IS ITS OWN STDLIB-ONLY MODULE, AND WHY IT SITS UNDER ``runtime``
-----------------------------------------------------------------------
``mobile/peer_send.py`` asks this question about every COLD target — an
address that resolved to a session with no live runtime — because a session
that has not been engaged yet must not be able to receive a peer message: a
row written there would become the opening row of a conversation its owner
never started. It may not import ``session/transcript.py`` to ask it. That
module reaches pydantic through ``harness.types``, and peer_send's import
weight is a PINNED contract — ``tests/unit/mobile/test_peer_send.py`` measures
it in a fresh interpreter and its AST guard forbids a ``local_operator.session``
import outside ``session.runtime`` (see the exemption note there). Choosing the
one dependency pair the guard already admits is what relocating this reader
does.

The alternative — re-implementing the row discriminator inside peer_send — is
worse than any package boundary, because two readers could then disagree about
what history IS, and an unengaged session would silently become a recipient
again the moment one of them drifted.

``session/transcript.py`` re-exports both names below, so there is still
exactly one definition of each and no existing caller had to move.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: Name of the transcript file inside a session directory. Defined here (and
#: re-exported by ``session/transcript.py``) because this module must not import
#: that one — see the module docstring.
TRANSCRIPT_FILENAME = "transcript.jsonl"

#: The two row fields that decide whether a row is history. Spelled out rather
#: than imported from ``session/transcript.py``, which this module exists to
#: avoid importing: these are FORMAT identifiers — ``type: message`` with a
#: ``payload.kind`` of ``message`` (or absent, for a row written before the
#: field existed). transcript.py owns the same two spellings for the writer and
#: the replay reader; a change to the journal format has to move all three, and
#: the round-trip tests there are what catch a partial move.
_MESSAGE_ENTRY_TYPE = "message"
_MESSAGE_KIND = "message"


def durable_conversation_path(path: Any) -> bool:
    """Whether the transcript file at ``path`` holds a REAL conversation turn.

    The seed signal for the record's ``started`` bit: ``RuntimeServer.__init__``
    (a resumed boot must publish ``started=True`` before any turn runs in the
    NEW process) and ``TuiSessionHandle.rebind`` (a ``/resume`` mid-flight
    re-seeds the bit for the swapped identity). A message row alone is NOT the
    discriminator: a round-1 quiet-dial of a peer note persists a
    ``peer_message`` CustomMessage as a message row (kind ``custom``) through
    ``append_messages``, without a turn ever running, so a session whose only
    durable rows are quiet-dial notes would seed ``started=True`` — and a
    peer's ``--wake`` or a broadcast would then drive an assistant turn into a
    session the owner never typed in (QA Q4). Only a plain ``Message`` row
    (kind ``message`` — a real user/assistant/tool turn) counts. Read from the
    FILE rather than the in-memory index — the index is built by replay and is
    not guaranteed populated at the moment either call site asks — and
    defensively: no readable file answers False, the conservative "unstarted"
    direction a first real turn immediately corrects.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for row in handle:
                try:
                    entry = json.loads(row)
                except ValueError:
                    continue  # a torn line says nothing about history
                if not isinstance(entry, dict) or entry.get("type") != _MESSAGE_ENTRY_TYPE:
                    continue
                payload = entry.get("payload")
                if not isinstance(payload, dict):
                    continue
                # ``kind`` arrived with producer admission; a legacy row
                # predating it IS a plain Message — the custom writer always
                # tagged its rows.
                if payload.get("kind", _MESSAGE_KIND) != _MESSAGE_KIND:
                    continue
                return True
    except OSError:
        return False
    return False


def session_has_durable_history(session_id: str, *, root: Path | str) -> bool:
    """Whether the session stored under ``root`` has been engaged yet.

    The by-ID face of :func:`durable_conversation_path`, for callers that hold a
    session ID rather than a ``Session`` (``RuntimeServer.has_durable_history``
    covers the object-shaped callers). ``root`` is the config dir — the
    directory holding ``sessions/`` — and is required rather than defaulted so a
    caller cannot reach for a global implicitly; peer_send's wrapper passes
    ``config_dir()``.

    NEVER raises: an absent session directory, an absent or unreadable
    transcript, and a truncated or corrupt file all answer False, the
    conservative unengaged direction a first real turn immediately corrects.
    """
    return durable_conversation_path(Path(root) / "sessions" / session_id / TRANSCRIPT_FILENAME)
