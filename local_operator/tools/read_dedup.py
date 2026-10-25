"""Read dedup: answer an unchanged, still-resident re-read with a pointer.

Why this exists
---------------
A ``read`` result is billed once when it lands and then RE-BILLED on every later
request of the session until compaction removes it. Measured across 7 days of
this operator's transcripts (2026-10, the manager's ``reread.py``): 14% of all
``read`` result bytes — 22.4 MB of 157.5 MB a week, the same ratio for every
role — were re-reads of a file span the same context had ALREADY fully viewed,
with no edit, write or mutating command in between and no compaction in
between. Each such re-read put a second copy of bytes the model could already
see into the prompt, and paid for that copy on every later call.

What it does
------------
The session owns one :class:`ReadLedger`. Every successful plain-file text read
records ``(realpath, effective line span, file fingerprint, tool_call_id)``.
A later read of the same realpath whose requested span is FULLY covered by a
ledger entry is answered with a short pointer instead of the content — but only
when all three hold at call time:

1. **Unchanged.** The file's sha256 now equals the one recorded when the earlier
   result was produced (size is a cheap pre-reject; the hash is the verdict —
   same size and a restored mtime with different bytes is a real case, e.g.
   ``git checkout`` of a same-length edit). This is why no edit/write hook is
   needed: our own mutating tools, a shell ``sed -i`` and an external editor
   are all caught the same way.
2. **Resident.** The earlier result message is still in the LIVE history the
   next request is built from, under the same ``tool_call_id``, not blanked
   (the ``pruned`` flag :func:`local_operator.compaction.pruning._blank` sets —
   supersede, useless and any later clearing pass that reuses ``_blank`` all
   honour it) and not an error. Residency is checked against the live list at call time —
   never cached — because compaction REPLACES that list: an entry whose
   result was summarised away simply stops being found, with no invalidation
   hook to forget.
3. **Not opted out.** ``raw=true`` bypasses dedup (see :data:`DEDUP_NOTICE`).

What it deliberately never touches
----------------------------------
Internal URLs (``skill://``, ``guide://``, ``spill://``, ``mcp://``, http,
``scratchpad://`` — the read tool only consults the ledger on its plain-file
path), errors, images, binaries, directory listings, Python structural
summaries and skill/guide files shaped by their on-disk path. A file over
:data:`MAX_FINGERPRINT_BYTES` is never recorded, because the verdict would cost
a full pass over a file the ranged reader deliberately never loads.

The span recorded is what the MODEL SEES, not what it asked for
---------------------------------------------------------------
:func:`visible_end` derives it from the rendered text as the wire will carry
it. That matters twice. A clamped read (8 KiB budget, or the line cap) shows
only a head of the requested span; and the replay bound then elides the middle
of that clamped result again, because body plus truncation footer exceeds
:data:`~local_operator.harness.replay_bound.DEFAULT_REPLAY_BOUND_CHARS`. Only
the contiguous head that survives both is recorded. The distinction carries
most of the measured number: on 400 real transcripts 91% of the bytes the
motivating "same span re-read" rule counted were re-requests whose earlier
read had been clamped — the model had NOT seen those lines, so no pointer may
claim it did. ``scripts/bench_read_dedup.py`` reports the figure this rule
actually recovers.

The supersede interaction (load-bearing)
----------------------------------------
The pointer result carries NO ``details['path']``. Compaction's supersede pass
(``pruning._supersede_key``) groups read results by ``details['path']`` and
blanks every earlier result with the same key — so a pointer carrying the path
would erase the very result it points at, leaving the model a dangling
reference and no content at all. The analytics marker rides under
``details['__dedup']`` instead, which no pruning pass reads.
"""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "DEDUP_DETAILS_KEY",
    "visible_end",
    "MAX_FINGERPRINT_BYTES",
    "FileFingerprint",
    "LedgerEntry",
    "ReadLedger",
    "dedup_notice",
    "fingerprint_file",
]

#: Analytics marker on a pointer result's details. Double-underscored so no
#: renderer or pruning pass mistakes it for a field it already reads.
DEDUP_DETAILS_KEY = "__dedup"

#: Largest file the ledger fingerprints. The whole-file read cap
#: (``READ_FILE_LIMIT_BYTES``, 2 MiB) on purpose: anything larger is only ever
#: read by range, and hashing it on every ranged read would reintroduce the
#: full pass the streaming reader exists to avoid.
MAX_FINGERPRINT_BYTES = 2 * 1024 * 1024

#: Bound on recorded entries per session. A ledger is in-memory bookkeeping
#: for one runtime; a long session paging through many files must not grow it
#: without limit. Oldest-recorded entries go first — they are also the ones
#: most likely compacted away already.
MAX_ENTRIES = 1024


@dataclass(frozen=True)
class FileFingerprint:
    """Identity of a file's bytes. ``size`` is the cheap reject; ``sha256``
    is the verdict. ``mtime_ns`` is kept for diagnostics only — a restored
    mtime with different bytes must still miss, so it never decides."""

    size: int
    mtime_ns: int
    sha256: str


@dataclass(frozen=True)
class LedgerEntry:
    """One recorded read: lines ``start..end`` (1-based, inclusive) of
    ``realpath`` as they were when ``fingerprint`` was taken. ``eof`` says the
    span reached the end of the file, so a later request running past ``end``
    asks for lines that do not exist and is still covered."""

    realpath: str
    start: int
    end: int
    eof: bool
    fingerprint: FileFingerprint
    tool_call_id: str
    #: Characters of the recorded result — what a pointer saves, for analytics.
    chars: int = 0

    def covers(self, start: int, end: int | None) -> bool:
        """Whether this entry shows every line of the request ``start..end``
        (``end=None`` = to EOF). A request starting past the recorded span
        never qualifies, even at EOF: its real answer is the "no lines in
        that window" notice, not a pointer to text that does not contain it.
        """
        if start < self.start or start > self.end:
            return False
        if end is None:
            return self.eof
        return end <= self.end or self.eof


def fingerprint_file(path: str | Path, data: bytes | None = None) -> FileFingerprint | None:
    """Fingerprint ``path``; ``None`` when it cannot be (gone, too large,
    unreadable). ``data`` is the bytes the caller already holds for exactly
    this read — passing them makes the fingerprint describe the SAME bytes
    the result was rendered from, with no second read to race."""
    try:
        stat = os.stat(path)
        if data is None:
            if stat.st_size > MAX_FINGERPRINT_BYTES:
                return None
            with open(path, "rb") as handle:
                data = handle.read(MAX_FINGERPRINT_BYTES + 1)
        if len(data) > MAX_FINGERPRINT_BYTES:
            return None
    except OSError:
        return None
    return FileFingerprint(
        size=len(data), mtime_ns=stat.st_mtime_ns, sha256=hashlib.sha256(data).hexdigest()
    )


#: Where a rendered ``read`` body stops being numbered lines: the clamp's
#: footer (``[truncated at …]``), the line-cap footer (``[N more lines …]``)
#: and the replay elision marker all open with a blank line and ``[``. Numbered
#: output can never contain that sequence — every rendered line, an empty file
#: line included, starts with its number.
_BODY_END = "\n\n["

#: ``tools.builtin.READ_OUTPUT_LIMIT_CHARS``, mirrored (importing the 30k-line
#: tool module from here would be a cycle in spirit) and pinned by a test.
READ_CLAMP_CHARS = 8 * 1024


def visible_end(text: str, start: int, replay_bound: int) -> int | None:
    """Last line number of the contiguous run ``start..N`` the model receives
    WHOLE from rendered ``read`` output ``text``, or ``None`` if not one line.

    The text is first passed through the replay bound exactly as every request
    will pass it (``harness.replay_bound``, deterministic), then cut at the
    first footer/marker. A trailing fragment that is not newline-terminated
    is a partially shown line — the head snap keeps whole lines, but one
    enormous line cannot be snapped — and does not count.
    """
    from local_operator.harness.replay_bound import REPLAY_RESULT_MARKER, _elide

    wire = text if len(text) <= replay_bound else _elide(text, replay_bound, REPLAY_RESULT_MARKER)
    cut = wire.find(_BODY_END)
    head = wire if cut == -1 else wire[:cut]
    if cut != -1 and wire.startswith(REPLAY_RESULT_MARKER.split("{elided}")[0], cut):
        # The replay head snaps to a line break, so a whole last line ends
        # with "\n"; without one it is the raw slice of an enormous line.
        head = head[:-1] if head.endswith("\n") else head.rpartition("\n")[0]
    elif cut != -1 and "\n" not in head and len(head) >= READ_CLAMP_CHARS:
        # ``_clamp_file_body`` cuts at the last line break, so its head is
        # whole lines — except when the first line alone overran the budget
        # and it kept the raw slice of it: nothing is visible whole.
        return None
    # Otherwise the tool's own footers follow a head cut at a line break.
    count = head.count("\n") + 1 if head else 0
    return start + count - 1 if count else None


def _is_blanked(message: Any) -> bool:
    """The ``pruned`` marker, read exactly where ``pruning._is_pruned`` reads
    it (payload, then details). Duplicated rather than imported: the harness
    tool layer must not import the compaction package, and the two
    definitions are pinned together by a test."""
    payload = getattr(message, "provider_payload", None)
    if not isinstance(payload, dict):
        return False
    if payload.get("pruned"):
        return True
    details = payload.get("details")
    return isinstance(details, dict) and bool(details.get("pruned"))


class ReadLedger:
    """Per-session record of plain-file reads, and the dedup verdict.

    ``live_messages`` returns the session's CURRENT LLM history — the list the
    next request is built from, not a snapshot. It is called per lookup
    because compaction replaces that list (see the module docstring).
    ``replay_bound`` is the replay bound the loop applies to every request;
    :func:`visible_end` uses it to record only what survives it.
    """

    def __init__(
        self,
        live_messages: Callable[[], Sequence[Any]],
        *,
        replay_bound: int,
        max_entries: int = MAX_ENTRIES,
    ) -> None:
        self._live_messages = live_messages
        self._replay_bound = replay_bound
        self._max_entries = max_entries
        self._entries: OrderedDict[tuple[str, str], LedgerEntry] = OrderedDict()

    @property
    def replay_bound(self) -> int:
        """The bound :func:`visible_end` must apply when recording."""
        return self._replay_bound

    def __len__(self) -> int:
        return len(self._entries)

    def record(self, entry: LedgerEntry) -> None:
        """Remember ``entry``; oldest entries beyond the cap are forgotten."""
        key = (entry.realpath, entry.tool_call_id)
        self._entries.pop(key, None)
        self._entries[key] = entry
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def candidates(self, realpath: str, start: int, end: int | None) -> list[LedgerEntry]:
        """Entries for ``realpath`` covering the request, newest first. Pure
        coverage — no residency, no disk — so the caller can skip the hash
        entirely when nothing covers."""
        return [
            entry
            for (path, _), entry in reversed(self._entries.items())
            if path == realpath and entry.covers(start, end)
        ]

    def resident(self, entry: LedgerEntry) -> bool:
        """Whether ``entry``'s result is in the live history, unblanked and
        sent whole."""
        for message in reversed(self._live_messages()):
            if getattr(message, "role", None) != "tool":
                continue
            if getattr(message, "tool_call_id", None) != entry.tool_call_id:
                continue
            # No length check here: the recorded span already is the part the
            # replay bound lets through (see :func:`visible_end`).
            ok = not getattr(message, "is_error", False) and not _is_blanked(message)
            if not ok:
                # Blanked is permanent for that message: forget it.
                self._entries.pop((entry.realpath, entry.tool_call_id), None)
            return ok
        # NOT found is deliberately not evicted: a read in the same parallel
        # batch as the one it repeats runs before the loop appends that batch's
        # results, so "absent" can mean "not yet landed". A compacted-away
        # entry simply keeps missing here and ages out under ``max_entries``.
        return False

    def find(
        self,
        realpath: str,
        start: int,
        end: int | None,
        current: Callable[[], FileFingerprint | None],
    ) -> LedgerEntry | None:
        """The resident, unchanged entry covering the request, or ``None``.

        ``current`` fingerprints the file NOW; it is called at most once, and
        only after a covering resident entry exists, so the common miss pays
        no disk I/O at all.
        """
        now: FileFingerprint | None = None
        fetched = False
        for entry in self.candidates(realpath, start, end):
            if not self.resident(entry):
                continue
            if not fetched:
                now, fetched = current(), True
            if now is None:
                return None
            if now.size == entry.fingerprint.size and now.sha256 == entry.fingerprint.sha256:
                return entry
        return None


def dedup_notice(display: str, start: int, end: int, tool_call_id: str) -> str:
    """The pointer text. Deterministic (same inputs, same bytes) so it never
    perturbs the provider prefix cache more than the call itself does.

    The escape hatch is ``raw=true``, reused rather than added. A new boolean
    on ``read`` is a schema tax on every request of every session (AGENTS.md,
    "The tool-surface footprint ladder"), while ``raw`` already means "give me
    the verbatim text rather than the default rendering"; a pointer is one more
    non-verbatim default, so ``raw=true`` bypassing it is the same promise,
    not a new one. The cost is that raw re-reads never dedup — measured by
    ``scripts/bench_read_dedup.py`` as a share of eligible re-reads.
    """
    short = tool_call_id[-8:] if len(tool_call_id) > 8 else tool_call_id
    return (
        f"[unchanged since your earlier read of {display} lines {start}-{end} "
        f"(call …{short}): that result is still above in your context — use it. "
        f"Pass raw=true to re-read anyway.]"
    )
