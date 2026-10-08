"""When the person last sent this conversation a message — for the RUNNING order.

WHY THIS EXISTS. The desktop sidebar's Running section orders rows by the time
of the person's own last message, not by the transcript's activity clock. The
two differ in exactly the way the operator reported: ``mtime`` (the wire's
activity time) advances when ANY row lands, so a response streaming in re-sorts
the section under the cursor — measured on the operator's store as 9 of 47
Running reorders under ``mtime`` over 235 s against 0 of 47 under this clock,
with ~3% of Running rows' ``mtime`` moving per 5 s poll. This module answers
"when did the person last send this session a message", which does not move
when the agent answers.

It is a READER, deliberately: nothing stamps and nothing writes. The transcript
is already the durable record of every send — a typed prompt, a steer, a
``lop send`` delivery — so the answer is derived on demand rather than threaded
through the three hosts that admit prompts (a write-through would also only
cover runtimes started AFTER the change, so a transcript reader is needed for
every session a build has ever written either way).

WHAT COUNTS AS THE PERSON'S OWN. A row counts when it is their words:
``type == "message"``, ``payload.role == "user"``, not a harness injection
(``payload.kind == "custom"`` — wake deliveries, peer/hub messages, job
results, session state, incidents and model-switch notices all ride that kind
and never count), and not harness chrome or a stamped render. That last
decision is NOT re-implemented here: ``transcript_index._is_harness_user_row``
is the one implementation every display fold already asks (the injection stamp,
the notice heads, the chrome prompt families), and this tracker asks the same
function so a row no surface may paint as the user's cannot order the Running
section as theirs. Two consequences are worth stating rather than discovering:

* A mid-turn STEER counts — it is a message the person sent — and its ``ts``
  is the DRAIN time: ``_drain_steering`` writes the row when the running turn
  next yields it, so the value trails the keystroke by however long the turn
  takes to reach a boundary. That lag is inherent to reading the transcript
  (one durable source of truth) and is accepted rather than hidden.
* A row whose ``ts`` cannot be read as a positive finite number is SKIPPED,
  and the scan keeps looking backward: reporting epoch zero would sort the
  row as infinitely old, which is a wrong claim where "unknown" (``None``) is
  an honest one.

THE COST MODEL, and why it is affordable on the 2 s poll. The scan is memoised
per session on ``(st_ino, st_size)`` and incremental: a COLD read walks
backward from EOF in 1 MiB chunks until it meets the newest qualifying row or
``COLD_SCAN_CAP_BYTES``; every later poll revalidates with ONE ``os.stat`` and
reads only the bytes appended since the last complete answer. Measured on the
operator's store (spike, 15 Running rows): cold 36 MB read / ~803 ms summed,
of which ~684 ms was the FIRST row — the lazy ``harness.rows`` import this
module deliberately does not move to import time; warm polls 0.1-2.3 ms for
all 15 rows. The cold cost is a ONE-TIME cost after a daemon start, paid in the
listing's worker thread, and it is cheaper than the uncached ``session_preview``
tail read the same listing already pays for every row (64 KB, uncached, every
poll). The module's own re-measurement is on the PR that introduced it.

THE MEMO is a process-local LRU of :data:`_MEMO_MAX` sessions, NOT pruned to
the live ids: a session that goes busy -> idle -> busy would otherwise pay a
cold scan again for the second busy window, while a 256-entry bound of one
path string plus a small tuple is bytes of headroom over anything a daemon
serves at once. It is guarded by a lock because the desktop listing runs on a
worker thread (``asyncio.to_thread``), and the COMPUTE runs outside the lock —
a duplicate scan on a race stores the same answer twice, while holding the
lock across a ~100 ms scan would queue every other session's row behind it.

READ-ONLY, ALWAYS: every open is ``O_RDONLY`` (``"rb"``), the only other
syscall is ``os.stat``, and nothing here writes, truncates or creates. The
inode is part of the memo key because ``Transcript.compact_file`` REPLACES the
journal with ``os.replace`` — a rewrite can keep the recorded size while no
byte at the tail can see the change — and a size SHRINK re-colds for the same
reason. A torn last line (no trailing newline — a writer mid-``fsync``) is
never consumed: the memo's covered offset stops at the last COMPLETE line, so
the finished row is picked up by the next poll instead of being half-read now.

UNITS. Epoch SECONDS as a float, the same unit as the wire's ``mtime`` and
``created_at``. ``None`` means UNKNOWN — no transcript, no typed row found
within the scan cap, or the file could not be read — and never "they never
sent anything".
"""

from __future__ import annotations

import json
import math
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import BinaryIO

from local_operator.session.runtime.engagement import TRANSCRIPT_FILENAME

#: The most bytes one COLD scan reads before giving up and answering ``None``.
#:
#: Sized against the store this feature was asked for: on the operator's store
#: the median distance from EOF to the newest typed row is 733 KB, p90 2.77 MB
#: and the observed maximum 12.2 MB, while transcripts reach ~100 MB — so ~32 MB
#: covers every observed live row with ~2.6x headroom over the worst one, and an
#: outlier answers "unknown" (the UI falls back to creation order) instead of
#: paying a 100 MB read on a 2 s poll. GIVING UP IS REMEMBERED, NOT RETRIED:
#: the memo records the size that was scanned, so a poll on the same bytes
#: costs one ``os.stat``; a later append still scans forward, so a typed row
#: that lands after the cap was hit is found.
COLD_SCAN_CAP_BYTES = 32 * 1024 * 1024

#: Read chunk for the backward walk. ``transcript_index._CHUNK_BYTES``'s own
#: figure, kept equal on purpose: both walk the same files with the same
#: granularity, and a slower chunk here would only make the cold scan pay more
#: syscalls per megabyte than the indexer beside it.
_CHUNK_BYTES = 1 << 20

#: How many sessions the memo remembers (LRU order on use).
_MEMO_MAX = 256

#: The byte sequence a role-user row's payload opens with, in the two spellings
#: a writer can produce: the compact separator (``TranscriptEntry.to_json`` and
#: everything the product writes) and the one-space form (hand-written or
#: tool-generated rows in fixtures). Used as a CHEAP REJECT only — megabytes of
#: tool output that carry neither sequence are never json-parsed — never as the
#: decision itself, which is the parsed row's.
_NEEDLE = b'"role":"user"'
_NEEDLE_SPACED = b'"role": "user"'

#: One memo entry: ``(inode, size, scanned_to, last_ts)`` where ``size`` is the
#: file size the entry was computed from, ``scanned_to`` is the offset up to
#: which COMPLETE lines have been examined (``<= size``; less than ``size`` when
#: a torn line was left unconsumed) and ``last_ts`` is the answer or ``None``.
_Entry = tuple[int, int, int, "float | None"]

#: ``{path: entry}`` in LRU order. Process-local, like every memo on this
#: module's import path (``model_selection._SELECTION_MEMO``'s reasoning), so a
#: daemon restart pays one cold scan per Running row and nothing is invalidated
#: on disk.
_MEMO: "OrderedDict[str, _Entry]" = OrderedDict()

#: Serialises memo hits, inserts and evictions only — see the module docstring
#: for why the compute stays outside it.
_MEMO_GUARD = threading.Lock()


def last_user_at(session_dir: Path) -> float | None:
    """When the person last sent this session a message, or ``None``.

    ``session_dir`` is the session's own directory; the journal inside it
    (``transcript.jsonl``) is the only file read. Never raises: a missing or
    unreadable transcript answers ``None``, as does a scan that hit
    :data:`COLD_SCAN_CAP_BYTES` without finding a typed row.
    """
    path = session_dir / TRANSCRIPT_FILENAME
    try:
        info = os.stat(path)
    except OSError:
        # Missing, or unstatable (a permission change mid-walk). Not memoised,
        # deliberately: a file that APPEARS later must be found, and a transient
        # failure must not freeze an answer under the path. Costs one failed
        # stat per call, which is the same shape every reader here degrades to.
        return None
    key = str(path)
    with _MEMO_GUARD:
        entry = _MEMO.get(key)
        if entry is not None and entry[0] == info.st_ino and entry[1] == info.st_size:
            _MEMO.move_to_end(key)
            return entry[3]
    try:
        with path.open("rb") as handle:
            scanned = _scan(path, handle)
    except OSError:
        # Exists but could not be READ — a fact about this moment, not about the
        # transcript. Answer, never memoise (a below-zero or chmod-torn state
        # heals itself; caching ``None`` under the path would not).
        return None
    with _MEMO_GUARD:
        _MEMO[key] = scanned
        _MEMO.move_to_end(key)
        while len(_MEMO) > _MEMO_MAX:
            _MEMO.popitem(last=False)
    return scanned[3]


def _scan(path: Path, handle: BinaryIO) -> _Entry:
    """The entry for the file behind ``handle``, cold or incremental.

    ``os.fstat`` on the OPEN handle is the key, not the earlier ``os.stat`` on
    the path: if the journal was replaced between the two, what this handle
    reads is what the key must describe, and a path stat taken before the open
    would describe the previous file.
    """
    st = os.fstat(handle.fileno())
    key = str(path)
    with _MEMO_GUARD:
        prev = _MEMO.get(key)
    if prev is not None and prev[0] == st.st_ino:
        if st.st_size == prev[1]:
            # The entry the lookup missed is still valid: it raced this call
            # (or was written between the path stat and the open). Serve it.
            return prev
        if st.st_size > prev[1]:
            return _scan_appended(handle, prev, st)
    return _scan_cold(handle, st)


def _scan_appended(handle: BinaryIO, prev: _Entry, st: os.stat_result) -> _Entry:
    """The bytes appended since the last complete answer, and nothing older.

    Reads from the memo's covered offset (not from the old SIZE), which is what
    completes a previously torn line exactly once: the re-read starts at that
    line's first byte, so the finished row parses whole. Only COMPLETE lines
    advance the covered offset — a new torn tail is left to the next poll.
    """
    scanned_to = prev[2]
    ts = prev[3]
    data = _read_range(handle, scanned_to, st.st_size)
    consumed = 0
    parts = data.split(b"\n")
    # ``parts[:-1]`` are the newline-terminated lines; the last element is an
    # unterminated tail (possibly empty) and is not consumed.
    for raw in parts[:-1]:
        consumed += len(raw) + 1
        candidate = _typed_ts(raw)
        if candidate is not None:
            ts = candidate
    return (st.st_ino, st.st_size, scanned_to + consumed, ts)


def _scan_cold(handle: BinaryIO, st: os.stat_result) -> _Entry:
    """Walk backward from EOF to the newest typed row, bounded by the cap.

    The carried fragment discipline is ``transcript_index._RowReader``'s,
    inverted: a chunk boundary can cut a line, so the piece before the first
    newline is carried into the next (earlier) chunk and completed by it. The
    one line that is never complete is the FILE's last, when the file does not
    end in a newline — it is skipped whole (see the module docstring), and the
    covered offset stops before it so the next poll consumes it once finished.
    """
    size = st.st_size
    if size == 0:
        return (st.st_ino, 0, 0, None)
    end = size
    carry = b""
    scanned_to: int | None = None
    #: Set once, from the first (tail) buffer: whether the file's last line
    #: lacks its terminating newline and must be skipped as mid-write.
    skip_tail = False
    read_total = 0
    while end > 0:
        if read_total >= COLD_SCAN_CAP_BYTES:
            # Over the cap: answer "unknown" and remember what was covered so
            # the same bytes are never rescanned (see COLD_SCAN_CAP_BYTES). The
            # covered offset is the resolved boundary when one is known — a
            # torn tail must still be re-read by the next append — else the
            # whole size, because the walk has given up on everything it read.
            return (st.st_ino, size, scanned_to if scanned_to is not None else size, None)
        start = max(0, end - _CHUNK_BYTES)
        buf = _read_range(handle, start, end) + carry
        read_total += end - start
        if end == size:
            skip_tail = not buf.endswith(b"\n")
        if scanned_to is None:
            # The FIRST newline met walking backward is the file's last, and
            # the bytes after it are the (possibly torn) tail. Its position is
            # what a later append must resume from, so it is the covered offset
            # whenever the tail is not newline-terminated.
            idx = buf.rfind(b"\n")
            if idx >= 0:
                scanned_to = start + idx + 1
        parts = buf.split(b"\n")
        if start > 0:
            # ``parts[0]`` ends at this buffer's first newline but does not
            # start at a line start: it is the tail of a line whose head is in
            # the next chunk back, carried until that chunk completes it.
            carry = parts[0]
            lines = parts[1:]
        else:
            carry = b""
            lines = parts
        for raw in reversed(lines):
            if not raw:
                continue
            if skip_tail:
                # The newest line has no trailing newline: mid-write, so not
                # consumed. Exactly one line can be in this state, and it is
                # the first non-empty one met walking backward.
                skip_tail = False
                continue
            candidate = _typed_ts(raw)
            if candidate is not None:
                return (
                    st.st_ino,
                    size,
                    scanned_to if scanned_to is not None else size,
                    candidate,
                )
        end = start
    # The head was reached without a newline ever appearing: a file with no
    # complete line at all has nothing covered (and nothing to hold on to).
    return (st.st_ino, size, scanned_to if scanned_to is not None else 0, None)


def _read_range(handle: BinaryIO, start: int, end: int) -> bytes:
    """One bounded read of ``[start, end)`` — the module's ONLY read site.

    A function rather than an inlined seek+read so tests can count the bytes a
    scan reads: the incremental property is asserted on reads, not on a clock
    (a timing assertion on a loaded machine measures contention).
    """
    handle.seek(start)
    return handle.read(end - start)


def _typed_ts(raw: bytes) -> float | None:
    """``raw`` is one complete journal line -> its ts if the person wrote it.

    The decision, in cost order: a byte pre-filter (an assistant or tool row
    is rejected without parsing), then the structural checks, then the SHARED
    chrome decision (``_is_harness_user_row``), which is imported lazily here
    so this module's own import graph stays stdlib + the transcript name.
    """
    if _NEEDLE not in raw and _NEEDLE_SPACED not in raw:
        return None
    try:
        entry = json.loads(raw)
    except ValueError:
        # Malformed rows are skipped individually, never fatal — the same
        # tolerance every reader of this journal carries (a live writer's
        # torn tail, a hand-edited file).
        return None
    if not isinstance(entry, dict) or entry.get("type") != "message":
        return None
    payload = entry.get("payload")
    if not isinstance(payload, dict) or payload.get("role") != "user":
        return None
    # ``kind == "custom"`` is every harness injection (wake, peer, hub,
    # job_result, session_state, session_incident, model switch — each carries
    # a ``custom_type``); an absent kind is a message row, the same reading
    # ``transcript._entry_to_message`` uses for rows written before the marker.
    if payload.get("kind") == "custom" or payload.get("custom_type"):
        return None
    ts = _usable_ts(entry.get("ts"))
    if ts is None:
        return None
    from local_operator.session.transcript_index import (
        _content_text,
        _is_harness_user_row,
    )

    if _is_harness_user_row(str(entry.get("id", "")), payload, _content_text(payload)):
        return None
    return ts


def _usable_ts(value: object) -> float | None:
    """A ts that can order a list, else ``None`` (and the row is skipped)."""
    try:
        ts = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(ts) or ts <= 0.0:
        return None
    return ts
