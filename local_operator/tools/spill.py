"""Content-addressed overflow store for oversized tool output.

Why this module exists
----------------------
A tool result has two jobs that pull in opposite directions: it must fit the
prompt, and it must not lose the answer. The old ``truncate_output`` resolved
that by keeping a head and a tail and replacing the middle with a marker — the
elided bytes were simply destroyed. An agent that needed line 2,400 of a
4,000-line pytest run had exactly two options, both worse than the truncation
was ever worth: guess, or re-run the command and pay the same truncated cost
again.

This store keeps the full text on disk under a content hash and hands the
model a ``spill://`` handle it can expand through the EXISTING ``read`` tool,
which already parses a line range. Nothing here adds a second tool or a second
URL convention; a handle is just another internal URL that ``read`` resolves,
and a range on it means what a range always means.

The bound is the point
----------------------
The reference implementation this borrows from (omp) caps what enters the
CONTEXT and spills the rest to disk, but retains those spills forever. On this
workstation that unbounded half grew ``~/.omp/agent/sessions`` to 6.8 GB with
single files up to 233 MB and filled the volume — which is the incident that
prompted this module. So the store here is bounded at BOTH ends:

- a per-entry cap, so one pathological command cannot claim the whole budget;
- a hard total-bytes ceiling enforced by LRU eviction on every write.

Neither bound is advisory. :meth:`SpillStore.write` evicts until the store
fits, and the ceiling holds even when a single session is the only writer.

Design decisions, stated so the next reader knows what is deliberate
--------------------------------------------------------------------
IMPLEMENTED:

- *Line-numbered addressing.* Entries are line-indexed and every read is
  rendered with 1-based line numbers, so a footer can point at a region and
  the model can ask for exactly that region back.
- *Search within a spill.* ``spill://<digest>?q=<regex>`` returns matching
  lines with their line numbers, so an agent hunting one traceback in 4,000
  lines finds the line number first and then reads 40 lines around it, rather
  than paging blindly through the blob at ~2k tokens a page.
- *Stable handles.* The handle is a content hash, so it is stable for the life
  of the entry, identical output written twice costs one entry, and a handle
  quoted in an older transcript still resolves as long as the entry survives.

DELIBERATELY LEFT OUT:

- *Context lines around a search match* (``grep -C``). Matches already carry
  line numbers and ``read`` already takes a range, so the second call gets the
  surroundings; a ``?c=`` parameter would be a third way to say the same thing.
- *Cross-entry search.* Searching every spill at once invites exactly the
  unbounded read this module exists to prevent.
- *A global index file.* Metadata lives in a per-entry sidecar. Tools run
  concurrently (``concurrency="shared"``), and a single shared index would
  need locking and could be corrupted by a crash mid-write; a directory scan
  on eviction is cheap at these entry counts and cannot desynchronize from the
  files it describes.

Every failure path here degrades rather than raises. Losing the ability to
expand an output is an inconvenience the footer can describe; raising into a
tool call is a bug that costs the whole turn.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from local_operator.paths import config_dir

__all__ = [
    "SPILL_DIRNAME",
    "SPILL_ENTRY_LIMIT_BYTES",
    "SPILL_MAX_BYTES_ENV",
    "SPILL_SCHEME",
    "SPILL_SEARCH_MATCH_LIMIT",
    "SPILL_SESSION_GRACE_MS",
    "SPILL_TOTAL_LIMIT_BYTES",
    "SpillMeta",
    "SpillRef",
    "SpillStore",
    "get_store",
    "parse_handle",
    "spill_dir",
]

#: URL scheme the ``read`` tool routes to this store. One scheme, reusing the
#: internal-URL path ``read`` already has for ``skill://``.
SPILL_SCHEME = "spill://"

#: Subdirectory under :func:`local_operator.paths.config_dir`. Inside the
#: config dir (rather than beside the logs) because the override
#: ``LOCAL_OPERATOR_CONFIG_DIR`` exists so a test or isolated run can be
#: certain the process touches ONE directory — a spill escaping to a real home
#: directory would break that promise and leave litter nothing cleans up.
SPILL_DIRNAME = "spill"

#: Hard ceiling on the whole store. 64 MiB is ~40x the entire measured
#: ``~/.local-operator`` footprint (1.6 MB) and ~1% of the 6.8 GB of unbounded
#: harness trajectories that actually filled this volume, so it is generous
#: for its purpose and still incapable of being the reason a disk fills. At
#: the 4 MiB per-entry cap it holds at least 16 worst-case entries, and at the
#: ~300 KB a full unit-suite pytest run measures, over 200 realistic ones.
SPILL_TOTAL_LIMIT_BYTES = 64 * 1024 * 1024

#: Environment override for the ceiling, for hosts that want a different
#: trade. Parsed leniently: a malformed value falls back to the default rather
#: than failing a tool call over a typo in an env var.
SPILL_MAX_BYTES_ENV = "LOCAL_OPERATOR_SPILL_MAX_BYTES"

#: Per-entry cap. Without it a single ``find /`` or a runaway build log would
#: evict every other entry in the store on its way in, which is the same
#: failure as having no store at all — the ceiling would hold while the
#: content the agent actually wanted disappeared. Oversized output is itself
#: spilled head-and-tail up to this cap and flagged ``complete=False``.
SPILL_ENTRY_LIMIT_BYTES = 4 * 1024 * 1024

#: Entries written by the CURRENT session within this window are evicted only
#: as a last resort. Without it, a turn that spills several large outputs can
#: evict the handle whose footer is still sitting unread in the live
#: transcript — the agent would follow a documented instruction to a dead
#: handle, which is worse than never having offered expansion.
SPILL_SESSION_GRACE_MS = 30 * 60 * 1000

#: Cap on matches returned by one ``?q=`` search. A search that matches every
#: line must not reintroduce the unbounded read the store exists to prevent.
SPILL_SEARCH_MATCH_LIMIT = 100

#: Length of the hex digest in a handle. 128 bits of SHA-256: collision-free
#: at any plausible entry count, and short enough that the handle in the
#: footer does not itself become a visual obstacle.
_DIGEST_CHARS = 32

#: Bounds the regex a model can hand the search path. A pathological pattern
#: costs CPU on text we deliberately keep large.
_MAX_QUERY_CHARS = 500

_HANDLE_RE = re.compile(r"^spill://([0-9a-f]{%d})(?:\?q=(.*))?$" % _DIGEST_CHARS, re.DOTALL)


@dataclass(frozen=True)
class SpillRef:
    """A parsed ``spill://`` handle: the entry, plus an optional search."""

    digest: str
    query: str = ""

    @property
    def handle(self) -> str:
        """The bare handle, without any query — what a footer should quote."""
        return f"{SPILL_SCHEME}{self.digest}"


@dataclass(frozen=True)
class SpillMeta:
    """Metadata about one stored entry. Never carries the content itself."""

    digest: str
    bytes: int
    lines: int
    complete: bool
    tool_name: str
    session_id: str
    created_ms: int
    last_read_ms: int

    @property
    def handle(self) -> str:
        return f"{SPILL_SCHEME}{self.digest}"


def parse_handle(text: str) -> SpillRef | None:
    """Parse a ``spill://`` handle, or ``None`` when ``text`` is not one.

    Returning ``None`` rather than raising is what lets ``read`` treat this as
    one branch of internal-URL dispatch: a string that is not a spill handle
    simply falls through to the next resolver.
    """
    match = _HANDLE_RE.match(text.strip())
    if match is None:
        return None
    return SpillRef(digest=match.group(1), query=(match.group(2) or "")[:_MAX_QUERY_CHARS])


def spill_dir() -> Path:
    """Directory holding the store. Resolved per call, never cached.

    ``config_dir()`` reads the environment on every call precisely so tests can
    relocate it after import; caching the joined path here would defeat that
    and let a test session write into a developer's real home directory.
    """
    return config_dir() / SPILL_DIRNAME


def _total_limit_bytes() -> int:
    """The ceiling in force, honouring :data:`SPILL_MAX_BYTES_ENV`."""
    raw = os.environ.get(SPILL_MAX_BYTES_ENV)
    if not raw:
        return SPILL_TOTAL_LIMIT_BYTES
    try:
        value = int(raw)
    except ValueError:
        return SPILL_TOTAL_LIMIT_BYTES
    # A zero or negative override would mean "evict everything you just
    # wrote", which is indistinguishable from a broken store; treat it as
    # unset rather than silently disabling expansion everywhere.
    return value if value > 0 else SPILL_TOTAL_LIMIT_BYTES


def _now_ms() -> int:
    return int(time.time() * 1000)


def _clip_to_entry_cap(text: str, cap: int) -> tuple[str, bool]:
    """``(text, complete)`` clipped to ``cap`` BYTES, head and tail.

    Head-and-tail rather than head-only for the same reason the in-context
    truncation keeps both ends: on a long build or test run the tail holds the
    summary and the failure, and a head-only clip of a 200 MB log stores 4 MB
    of banner and throws the answer away.

    The cap is a byte budget but the slicing is by characters, so the result
    is re-encoded and trimmed until it fits — a multi-byte character must not
    be able to push the entry past the cap or be split down the middle.
    """
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= cap:
        return text, True
    # Work in characters, then verify in bytes: a purely byte-based slice can
    # land mid-codepoint, and decoding that back produces replacement noise in
    # the middle of the stored text.
    half = cap // 2
    head_chars = half
    tail_chars = cap - half
    while True:
        head = text[:head_chars]
        tail = text[len(text) - tail_chars :]
        joined = head + tail
        if len(joined.encode("utf-8", errors="replace")) <= cap:
            return joined, False
        # Shrink proportionally; the loop is bounded because both counts fall
        # monotonically and the empty string always fits.
        head_chars = head_chars * 9 // 10
        tail_chars = tail_chars * 9 // 10
        if head_chars == 0 and tail_chars == 0:
            return "", False


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically replace ``path`` through a unique same-directory temp.

    Same-directory keeps ``os.replace`` atomic. A unique name is load-bearing
    now that oversized tool results spill from worker threads: two identical
    outputs share the same content digest, so the old deterministic
    ``<digest>.txt.tmp`` let one writer rename the other's temp away.
    """
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            # Record the name BEFORE the first byte. A full disk can raise
            # from `write`; the finally block must still remove the partial
            # file, which is invisible to the store's entry accounting.
            tmp_path = Path(stream.name)
            stream.write(data)
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


class SpillStore:
    """Bounded, content-addressed store of full tool outputs.

    One instance per config directory; :func:`get_store` hands out the
    process-wide one. Every public method degrades on I/O failure — a store
    that cannot be written must not be able to fail a tool call.
    """

    __slots__ = ("_root", "_write_lock")

    def __init__(self, root: Path | None = None) -> None:
        # ``None`` means "resolve per call", so an instance built before a test
        # relocated the config dir still follows the relocation.
        self._root = root
        # One write transaction is content + sidecar + eviction. Worker-thread
        # spilling made those phases concurrent across tools; serializing the
        # whole unit prevents one writer's sweep from deleting another's
        # half-installed or just-installed digest. Oversized spills are rare,
        # so one store lock is simpler and safer than per-digest lock lifetime.
        self._write_lock = threading.Lock()

    @property
    def root(self) -> Path:
        return self._root if self._root is not None else spill_dir()

    # -- paths -------------------------------------------------------------

    def _content_path(self, digest: str) -> Path:
        return self.root / f"{digest}.txt"

    def _meta_path(self, digest: str) -> Path:
        return self.root / f"{digest}.json"

    # -- write -------------------------------------------------------------

    def write(self, text: str, *, tool_name: str = "", session_id: str = "") -> SpillMeta | None:
        """Store one output as an install+evict transaction.

        The lock covers both atomic entry files and the eviction sweep: a
        concurrent writer's digest cannot be deleted during its
        content-before-sidecar interval or immediately after installation.
        """
        with self._write_lock:
            return self._write_locked(text, tool_name=tool_name, session_id=session_id)

    def _write_locked(
        self, text: str, *, tool_name: str = "", session_id: str = ""
    ) -> SpillMeta | None:
        """Store ``text`` and return its metadata, or ``None`` on failure.

        ``None`` is a normal outcome, not an error: a read-only home
        directory, a full disk, or a sandbox that forbids the write all land
        here, and the caller's contract is to fall back to plain truncation.
        Raising instead would turn a degraded expansion path into a failed
        tool call, which is strictly worse than the truncation it replaced.

        Writing the same text twice is idempotent — the digest is the identity
        — and refreshes the entry's recency so a repeated command does not
        make its own handle the next eviction victim.
        """
        if not text:
            return None
        try:
            clipped, complete = _clip_to_entry_cap(text, SPILL_ENTRY_LIMIT_BYTES)
            if not clipped:
                return None
            data = clipped.encode("utf-8", errors="replace")
            digest = hashlib.sha256(data).hexdigest()[:_DIGEST_CHARS]
            root = self.root
            root.mkdir(parents=True, exist_ok=True, mode=0o700)

            now = _now_ms()
            meta = SpillMeta(
                digest=digest,
                bytes=len(data),
                lines=clipped.count("\n") + 1,
                complete=complete,
                tool_name=tool_name,
                session_id=session_id,
                created_ms=now,
                last_read_ms=now,
            )
            self._write_entry(digest, data, meta)
            # Evict AFTER the write, never before: sizing the eviction against
            # a prediction of this entry's size would drift from the truth for
            # multi-byte text, and evicting first cannot protect a write that
            # then fails anyway.
            self._evict(session_id=session_id, keep=digest)
            return meta
        except OSError:
            return None

    def _write_entry(self, digest: str, data: bytes, meta: SpillMeta) -> None:
        """Write content and sidecar atomically through unique temp names.

        Content first, then metadata: content without a sidecar is invisible
        to :meth:`stat` and swept by eviction, while metadata pointing at
        absent content would create a handle that resolves to nothing. Both
        files use unique same-directory temps so concurrent writers of the
        same content digest cannot rename one another's staging file away or
        interleave a sidecar.
        """
        _atomic_write_bytes(self._content_path(digest), data)
        encoded_meta = json.dumps(
            {
                "digest": meta.digest,
                "bytes": meta.bytes,
                "lines": meta.lines,
                "complete": meta.complete,
                "tool_name": meta.tool_name,
                "session_id": meta.session_id,
                "created_ms": meta.created_ms,
                "last_read_ms": meta.last_read_ms,
            }
        ).encode("utf-8")
        _atomic_write_bytes(self._meta_path(digest), encoded_meta)

    # -- read --------------------------------------------------------------

    def stat(self, handle: str) -> SpillMeta | None:
        """Metadata for ``handle`` without reading its content.

        ``None`` means the handle is unknown or was evicted — which callers
        must treat as ordinary, because a bounded store guarantees it happens.
        """
        ref = parse_handle(handle) if handle.startswith(SPILL_SCHEME) else SpillRef(handle)
        if ref is None:
            return None
        return self._load_meta(ref.digest)

    def _load_meta(self, digest: str) -> SpillMeta | None:
        try:
            raw = json.loads(self._meta_path(digest).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(raw, dict):
            return None
        if not self._content_path(digest).exists():
            return None
        try:
            return SpillMeta(
                digest=str(raw.get("digest", digest)),
                bytes=int(raw.get("bytes", 0)),
                lines=int(raw.get("lines", 0)),
                complete=bool(raw.get("complete", True)),
                tool_name=str(raw.get("tool_name", "")),
                session_id=str(raw.get("session_id", "")),
                created_ms=int(raw.get("created_ms", 0)),
                last_read_ms=int(raw.get("last_read_ms", 0)),
            )
        except (TypeError, ValueError):
            return None

    def _load_text(self, digest: str) -> str | None:
        try:
            return self._content_path(digest).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    def _touch(self, digest: str) -> None:
        """Record a read so LRU eviction keeps what is actually being used.

        Best-effort: the sidecar mtime is the recency signal eviction sorts on,
        so a failure here only makes an entry look staler than it is.
        """
        try:
            os.utime(self._meta_path(digest), None)
        except OSError:
            return

    def read_lines(
        self, handle: str, start: int = 1, end: int | None = None
    ) -> tuple[list[str], int] | None:
        """``(selected_lines, total_lines)`` for a 1-based inclusive range.

        ``None`` when the handle is unknown. An out-of-range request returns an
        empty list with the true total, so the caller can tell the model how
        many lines there actually are instead of just failing.
        """
        ref = parse_handle(handle) if handle.startswith(SPILL_SCHEME) else SpillRef(handle)
        if ref is None:
            return None
        text = self._load_text(ref.digest)
        if text is None:
            return None
        self._touch(ref.digest)
        lines = text.splitlines()
        return lines[max(start, 1) - 1 : end], len(lines)

    def search(
        self, handle: str, query: str, limit: int = SPILL_SEARCH_MATCH_LIMIT
    ) -> tuple[list[tuple[int, str]], int, int] | None:
        """``(matches, total_matches, total_lines)`` for a regex over an entry.

        Each match is ``(line_number, line_text)`` so the model can follow up
        with a range read around the hit. ``total_matches`` counts every match,
        not just the returned ones, so a truncated match list can say so.
        ``None`` when the handle is unknown; an invalid regex raises
        ``re.error`` for the caller to render as a correctable message.
        """
        ref = parse_handle(handle) if handle.startswith(SPILL_SCHEME) else SpillRef(handle)
        if ref is None:
            return None
        text = self._load_text(ref.digest)
        if text is None:
            return None
        self._touch(ref.digest)
        regex = re.compile(query[:_MAX_QUERY_CHARS])
        lines = text.splitlines()
        matches: list[tuple[int, str]] = []
        total = 0
        for number, line in enumerate(lines, start=1):
            if regex.search(line):
                total += 1
                if len(matches) < limit:
                    matches.append((number, line))
        return matches, total, len(lines)

    # -- bookkeeping -------------------------------------------------------

    def _entries(self) -> list[tuple[str, int, float, str]]:
        """``(digest, bytes, recency, session_id)`` for every readable entry.

        Recency is the sidecar mtime, which :meth:`_touch` bumps on read, so
        the ordering is genuinely least-recently-USED and not merely oldest-
        written. Unreadable or orphaned files are reported with a recency of
        0.0 so eviction sweeps them first.
        """
        entries: list[tuple[str, int, float, str]] = []
        try:
            names = list(self.root.iterdir())
        except OSError:
            return entries
        for path in names:
            if path.suffix != ".txt":
                continue
            digest = path.stem
            try:
                size = path.stat().st_size
            except OSError:
                continue
            meta_path = self._meta_path(digest)
            try:
                recency = meta_path.stat().st_mtime
            except OSError:
                # Content with no sidecar: a crash between the two writes, or
                # a swept sidecar. Unreadable by stat(), so it is pure waste
                # and must go first.
                entries.append((digest, size, 0.0, ""))
                continue
            meta = self._load_meta(digest)
            entries.append((digest, size, recency, meta.session_id if meta else ""))
        return entries

    def total_bytes(self) -> int:
        """Bytes of stored content (sidecars excluded — they are ~200 B each
        and counting them would make the ceiling depend on entry count)."""
        return sum(size for _digest, size, _recency, _session in self._entries())

    def entry_count(self) -> int:
        return len(self._entries())

    def _remove(self, digest: str) -> int:
        """Delete one entry; returns the bytes reclaimed."""
        reclaimed = 0
        for path in (self._content_path(digest), self._meta_path(digest)):
            try:
                reclaimed += path.stat().st_size if path.suffix == ".txt" else 0
                path.unlink()
            except OSError:
                continue
        return reclaimed

    def _evict(self, *, session_id: str = "", keep: str = "") -> int:
        """Evict least-recently-used entries until the store fits the ceiling.

        Two passes, because a hard ceiling and a usable live session are both
        requirements and they conflict at the margin:

        1. Evict LRU among entries that are NOT recent writes of the calling
           session. This is the normal path and it never touches a handle the
           current turn just published.
        2. If the store still does not fit — a single session outrunning the
           whole budget by itself — evict LRU among the protected ones too.
           The ceiling is not negotiable; the grace window is.

        ``keep`` is the entry just written, which is never evicted by its own
        write: returning a handle that was deleted on the way out would be a
        footer pointing at nothing.

        Returns the number of entries evicted.
        """
        limit = _total_limit_bytes()
        entries = self._entries()
        total = sum(size for _digest, size, _recency, _session in entries)
        if total <= limit:
            return 0

        now_seconds = time.time()
        grace_seconds = SPILL_SESSION_GRACE_MS / 1000.0

        def protected(digest: str, recency: float, owner: str) -> bool:
            if digest == keep:
                return True
            if not session_id or owner != session_id:
                return False
            return (now_seconds - recency) < grace_seconds

        evicted = 0
        for allow_protected in (False, True):
            if total <= limit:
                break
            candidates = [
                (recency, digest, size)
                for digest, size, recency, owner in entries
                # ``keep`` is excluded from BOTH passes: the entry this write
                # just created must survive its own eviction sweep.
                if digest != keep and (allow_protected or not protected(digest, recency, owner))
            ]
            candidates.sort()  # oldest recency first
            for _recency, digest, size in candidates:
                if total <= limit:
                    break
                self._remove(digest)
                total -= size
                evicted += 1
            entries = [entry for entry in entries if self._content_path(entry[0]).exists()]
        return evicted

    def prune_all(self) -> int:
        """Delete every entry as one transaction; returns the count removed."""
        with self._write_lock:
            return self._prune_all_locked()

    def _prune_all_locked(self) -> int:
        """Delete every entry. Returns the count removed (tests, teardown)."""
        removed = 0
        for digest, _size, _recency, _session in self._entries():
            self._remove(digest)
            removed += 1
        return removed


#: Process-wide store. Built with ``root=None`` so it re-resolves the config
#: directory on every access rather than freezing whatever the first importer
#: saw — the same reason ``config_dir()`` reads the environment per call.
_STORE = SpillStore()


def get_store() -> SpillStore:
    """The process-wide store."""
    return _STORE
