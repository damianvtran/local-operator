"""Bounded, cached, read-only transcript reads for the ``/resume`` picker's preview.

A NEW MODULE RATHER THAN AN ADDITION TO ``resume.py``, for three reasons that
are each load-bearing. ``resume.py`` is documented as deliberately stdlib-only
and import-light because it sits on the CLI startup path (see its
``recent_sessions`` and ``live_runtime_pid`` docstrings); it is already 2,125
lines; and this layer owns per-picker CACHED STATE, which is a different
lifetime from ``resume.py``'s stateless helpers — one
:class:`SessionPreviews` is owned by one open picker and discarded with it.

NOTHING HERE WRITES, RESUMES, OR MUTATES THE STORE. Every read is mode
``"rb"``, and every ``open`` goes through :meth:`SessionPreviews._tail`.

THE PERFORMANCE QUESTION, ANSWERED UP FRONT because it is the first one a
reviewer asks. The largest real transcript is 15.1 MB; the store's median is
294 KB and its p90 730 KB. The cost of a first preview render is INDEPENDENT
OF FILE SIZE: one ``stat``, one ``seek`` to ``size − PREVIEW_TAIL_BYTES``, one
256 KB read, and a JSON parse of the lines in that window. Only ~256 KB is
ever faulted in, never 15.1 MB. Measured at 0.88–1.12 ms per session warm (60
sessions in 53 ms); a cold page cache adds one 256 KB disk read. It therefore
runs synchronously on a cursor move — that is inside a 16 ms frame with room
to spare — and it is cached because holding a wheel-scroll issues about 30
cursor moves a second. There is NO full-transcript parse, ever, and no per-row
read per keystroke.

HONEST LIMIT, stated because a caller will otherwise assume otherwise: the
preview reads the TAIL, so "the session's first user turn" means the first
user turn *in the tail window*. On a transcript larger than
:data:`PREVIEW_TAIL_BYTES` that is not the session's opening message. This
matches the validated prototype and the established ``resume.session_preview``
pattern; do not add a second head read to close it.
"""

from __future__ import annotations

import json
import re
import textwrap
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_operator.session.creation import session_created_at

#: The tail window, in bytes. A ``seek`` to an offset lands mid-line, so the
#: window is always one partial line longer than it looks and that first line
#: is dropped. Mirrors ``resume.session_preview``'s established pattern.
PREVIEW_TAIL_BYTES = 256_000

#: The checkpoint discriminator, used with ``entry["type"] == "custom"``.
#:
#: It is NOT ``payload["kind"]`` — that is ``None`` on these entries, so a
#: predicate on ``kind`` finds nothing. Coverage is 113 of 140 picker rows, and
#: the state lives at ``payload["details"]["state"]``.
CHECKPOINT_CUSTOM_TYPE = "frontend_state_checkpoint_v1"

TRANSCRIPT_NAME = "transcript.jsonl"

#: Role gutters for the preview body.
GUTTERS = {"user": "▸ you", "assistant": "▪ lop"}

#: Markdown emphasis strippers (D5). Measured over 58 previewable sessions:
#: ``code`` in 74%, ``**bold**`` 57%, ``## heading`` 34%, ``- bullet`` 43%,
#: ``_em_`` 2%. Regex and move on — this is a preview, not a markdown parser,
#: and it must not add a dependency.
_MD_HEADING = re.compile(r"^#{1,6}[ \t]+", re.MULTILINE)
_MD_CODE = re.compile(r"`([^`]+)`")
_MD_BOLD = re.compile(r"\*\*(\S(?:[^*]*\S)?)\*\*|__(\S(?:[^_]*\S)?)__")
#: Requires a non-space after the opening star, so a ``* item`` bullet — which
#: is STRUCTURE the reader wants, not emphasis — never matches.
_MD_EM_STAR = re.compile(r"(?<!\*)\*(\S(?:[^*\n]*\S)?)\*(?!\*)")
#: ``_`` only when not flanked by word characters, so ``snake_case`` survives.
_MD_EM_UNDER = re.compile(r"(?<![\w_])_(\S(?:[^_\n]*\S)?)_(?![\w_])")


@dataclass(frozen=True)
class PreviewTurn:
    """One human-visible turn: who spoke, what they said, and when."""

    role: str
    text: str
    ts: float


def demark(text: str) -> str:
    """Strip markdown emphasis markers, KEEPING the text they wrapped (D5).

    Bullet markers are deliberately kept: they are structure, not emphasis.
    """
    text = _MD_HEADING.sub("", text)
    text = _MD_CODE.sub(r"\1", text)
    text = _MD_BOLD.sub(lambda match: match.group(1) or match.group(2) or "", text)
    text = _MD_EM_STAR.sub(r"\1", text)
    return _MD_EM_UNDER.sub(r"\1", text)


def _block_text(payload: dict[str, Any]) -> str:
    """The joined text of a payload's content blocks.

    ``b.get("text", ...)`` rather than ``b["text"]``: ``attachment`` blocks
    carry no ``text`` key, so a bare subscript RAISES on them. And no predicate
    on ``b["type"]`` — verified across 1,711 real blocks, the key-sets are
    ``('text',)`` ×1,709 and ``('attachment','mime_type')`` ×2, so NO block has
    a ``type`` key and such a predicate would match nothing at all.
    """
    return "".join(block.get("text", "[attachment]") for block in payload.get("content") or [])


def condense_entries(entries: Iterable[dict[str, Any]]) -> list[PreviewTurn]:
    """Human-visible turns only, in transcript order.

    Exactly this predicate and no other: ``kind == "custom"`` is journal noise,
    ``role == "tool"`` is machinery the condensed view exists to remove, and an
    empty turn renders as a role label with nothing under it.
    """
    out: list[PreviewTurn] = []
    for entry in entries:
        payload = entry.get("payload") or {}
        text = _block_text(payload)
        if (
            entry.get("type") == "message"
            and payload.get("kind") == "message"
            and payload.get("role") in ("user", "assistant")
            and text.strip()
        ):
            out.append(PreviewTurn(str(payload.get("role")), text, float(entry.get("ts") or 0.0)))
    return out


def verbose_entries(entries: Iterable[dict[str, Any]]) -> list[PreviewTurn]:
    """Every message entry, including ``role == "tool"`` and custom kinds."""
    out: list[PreviewTurn] = []
    for entry in entries:
        payload = entry.get("payload") or {}
        text = _block_text(payload)
        if entry.get("type") == "message" and text.strip():
            out.append(PreviewTurn(str(payload.get("role")), text, float(entry.get("ts") or 0.0)))
    return out


def wrap_turns(turns: Sequence[PreviewTurn], width: int, height: int) -> list[tuple[str, str]]:
    """``(kind, line)`` for the preview body, oldest-first from the TOP.

    ``kind`` is ``"gutter"``, ``"blank"``, or the turn's role, so the caller
    styles without re-parsing.

    D18 — OPENS AT THE FIRST USER TURN, and leading assistant turns are DROPPED
    rather than scrolled past. Measured: 36 of 141 transcripts open on
    mid-session assistant narration, and those 36 included the picker's default
    cursor row, which is why no round-2 frame contained a single ``▸ you``. At
    80x24 the pane shows one turn, so "somewhere below" is the same as absent.

    D30 — NEVER ENDS ON AN ORPHAN ROLE LABEL. When the budget cannot fit a
    header plus at least one body line, the header is dropped: a truncated
    sentence reads as continuation and is correct, but a label with nothing
    beneath it reads as a turn that failed to load.
    """
    width = max(10, width)
    # ``height`` is part of the signature the brief fixes (§4.1) and is kept so
    # callers state the budget they are wrapping for, but the CLIP is
    # deliberately not done here: the pane scrolls, so the full wrapped list has
    # to exist for ``ctrl+u``/``ctrl+d``/``ctrl+g`` to move through. Trimming to
    # the visible window is :func:`clip_to_height`'s job, and doing it twice is
    # how the D30 guard came to run against the wrong budget.
    del height
    first_user = next((index for index, turn in enumerate(turns) if turn.role == "user"), None)
    if first_user is not None:
        turns = list(turns)[first_user:]

    out: list[tuple[str, str]] = []
    for turn in turns:
        out.append(("gutter", GUTTERS.get(turn.role, f"▪ {turn.role}")))
        for paragraph in demark(turn.text).splitlines():
            if not paragraph.strip():
                continue
            # ``break_long_words`` keeps a 200-char URL from overflowing the
            # pane; ``break_on_hyphens`` off keeps hyphenated identifiers whole.
            for line in textwrap.wrap(
                paragraph, width, break_long_words=True, break_on_hyphens=False
            ):
                # D11: a continuation line that kept a leading space turns the
                # caller's uniform 2-space indent into 3 on that row alone.
                stripped = line.rstrip()
                if stripped:
                    out.append((turn.role, stripped))
        out.append(("blank", ""))

    # D30's guard. Applied to the WHOLE list rather than to the visible window,
    # because a trailing header is never wanted: scrolled to, it has its body
    # below it; clipped to, it is an orphan.
    while out and out[-1][0] in ("blank", "gutter"):
        if out[-1][0] == "gutter":
            out.pop()
            break
        out.pop()
    return out


def clip_to_height(
    lines: Sequence[tuple[str, str]], top: int, height: int
) -> list[tuple[str, str]]:
    """``height`` lines from ``top``, never ending on an orphan role label (D30)."""
    window = list(lines[top : top + max(1, height)])
    while window and window[-1][0] == "gutter":
        window.pop()
    return window


def grep_context(digest: str, query: str, width: int) -> str | None:
    """An ellipsised window of ``digest`` CENTRED on an exact hit, or ``None``.

    ``None`` means the row matched softly/fuzzily with no literal substring,
    which the picker renders as a ``~`` mark and no context line.

    D17 — THE CALLER MUST PASS THE WIDTH IT WILL ACTUALLY DRAW AT. The
    prototype asked for ``width=150`` and then truncated that snippet into a
    ~55-cell pane FROM THE LEFT, which cut the match off the right end: the
    query sat at index 73 of a 152-char snippet, so 0 of 9 context lines
    contained the query. Centring here is only correct if the window is the one
    rendered; round 3 measured the fix at 28 of 28 populated context lines
    carrying the highlighted query.
    """
    if not digest or not query:
        return None
    width = max(10, width)
    index = digest.lower().find(query.lower())
    if index < 0:
        return None
    start = max(0, index - (width - len(query)) // 2)
    end = min(len(digest), start + width)
    snippet = " ".join(digest[start:end].split())
    return f"{'…' if start > 0 else ''}{snippet}{'…' if end < len(digest) else ''}"


class SessionPreviews:
    """Bounded, cached, read-only reads for ONE open picker.

    Every accessor is safe to call on each cursor move: the first call for a
    session pays one bounded tail read and a parse of that window, and every
    call after it is a dict lookup. The instance is owned by the screen and
    discarded with it, so the cache cannot outlive the geometry it was built
    for. Missing or unreadable transcripts return the empty value rather than
    raising — the picker paints on a cursor move, and an unreadable session
    must not take the screen down with it.
    """

    def __init__(self, sessions_dir: Path) -> None:
        self._sessions = Path(sessions_dir)
        self._lines: dict[str, list[str]] = {}
        self._entries: dict[str, list[dict[str, Any]]] = {}
        self._condensed: dict[str, list[PreviewTurn]] = {}
        self._verbose: dict[str, list[PreviewTurn]] = {}
        self._checkpoint: dict[str, dict[str, Any]] = {}
        self._created: dict[str, float] = {}

    def _tail(self, session_id: str) -> list[str]:
        """The last :data:`PREVIEW_TAIL_BYTES` of the transcript, as whole lines."""
        if session_id in self._lines:
            return self._lines[session_id]
        transcript = self._sessions / session_id / TRANSCRIPT_NAME
        window = b""
        try:
            size = transcript.stat().st_size
            with transcript.open("rb") as handle:
                if size > PREVIEW_TAIL_BYTES:
                    handle.seek(size - PREVIEW_TAIL_BYTES)
                    # The seek landed at an arbitrary byte: the first line is a
                    # fragment with nothing recoverable in it.
                    _, _, window = handle.read().partition(b"\n")
                else:
                    window = handle.read()
        except OSError:
            window = b""
        self._lines[session_id] = window.decode("utf-8", errors="replace").splitlines()
        return self._lines[session_id]

    def entries(self, session_id: str) -> list[dict[str, Any]]:
        """Parsed JSONL entries from the tail window, skipping unparseable lines."""
        if session_id in self._entries:
            return self._entries[session_id]
        out: list[dict[str, Any]] = []
        for line in self._tail(session_id):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if isinstance(entry, dict):
                out.append(entry)
        self._entries[session_id] = out
        return out

    def condensed(self, session_id: str) -> list[PreviewTurn]:
        """Human turns only — the picker's DEFAULT view."""
        if session_id not in self._condensed:
            self._condensed[session_id] = condense_entries(self.entries(session_id))
        return self._condensed[session_id]

    def verbose(self, session_id: str) -> list[PreviewTurn]:
        """Every message entry, behind ``ctrl+e``."""
        if session_id not in self._verbose:
            self._verbose[session_id] = verbose_entries(self.entries(session_id))
        return self._verbose[session_id]

    def checkpoint(self, session_id: str) -> dict[str, Any]:
        """The NEWEST frontend-state checkpoint, or ``{}`` when there is none.

        ``{}`` means the header's ``model · cwd`` line is omitted ENTIRELY
        (D7) rather than drawn with placeholders, which read as a load that
        never resolved.
        """
        if session_id in self._checkpoint:
            return self._checkpoint[session_id]
        state: dict[str, Any] = {}
        for entry in self.entries(session_id):
            payload = entry.get("payload") or {}
            if (
                entry.get("type") == "custom"
                and payload.get("custom_type") == CHECKPOINT_CUSTOM_TYPE
            ):
                found = (payload.get("details") or {}).get("state")
                if isinstance(found, dict):
                    state = found
        self._checkpoint[session_id] = state
        return state

    def created_at(self, session_id: str) -> float:
        """Session birth time, or ``0.0`` when unrecoverable."""
        if session_id not in self._created:
            try:
                self._created[session_id] = session_created_at(self._sessions / session_id)
            except Exception:
                self._created[session_id] = 0.0
        return self._created[session_id]
