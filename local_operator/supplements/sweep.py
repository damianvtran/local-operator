"""The turn-supplement sweep: reclaim orphaned component blobs, settle stale rows.

WHAT IT CLEANS, AND WHY EACH HALF IS SAFE.

**Orphaned blobs.** A generated component lives in the shared content-addressed
attachment store (``session/attachments.py``) under the digest its row carries.
The store's own contract is *nothing is ever deleted* -- an attachment lives for
as long as any transcript references it -- and that contract is exactly what
this module does NOT touch: it never deletes a blob any journal still
references, never touches a blob whose sidecar mime is not ``text/html`` (the
mime ONLY supplements write; every media producer goes through ``cache_media``,
which refuses non-media kinds), and never deletes anything younger than the
grace window. What it reclaims is the narrow class the store's contract cannot:
a component blob stored by a job that then died before its terminal row landed
(the row write is the only way a blob becomes referenced). Such a blob is
unreachable forever -- the document route authorises by journal reference -- so
reclaiming it is reclaiming bytes the user can never see, not deleting work.

The grace window is the in-flight guarantee. A job's whole lifetime is bounded
by ``supplements.timeoutS`` (90 s default) plus one row append; the window is
:data:`SWEEP_GRACE_S` (1 hour, 40x that bound), so a blob a LIVE job is about
to reference cannot be a candidate even if the sweep ran at the worst possible
moment. The sweep is additionally interval-gated, runs off the critical path
(never on the turn path), and is bounded per pass.

**Stale rows.** A non-terminal journal row (``decided``/``queued``) can only
belong to a job whose runtime died: at the moment a runtime's runner is built
nothing has run yet, and a live job's own progress is tracked in memory. The
frozen reader rule already renders such a row as "cancelled - Retry" without
any write; settling it here makes the journal tell the same story the reader
does -- the job is over -- while keeping the audit trail (older versions stay).

WHAT THIS MODULE IS NOT. Not a retention policy for user media (the store's
never-delete contract stands, untouched), not a sync/copy path (nothing here
moves bytes anywhere), and not a route-time authorisation (the routes read the
journal themselves). It is one bounded, best-effort maintenance pass.

BOUNDED, IN THREE LAYERS.
- Per pass: at most :data:`MAX_CANDIDATES` blobs examined (oldest first), at
  most :data:`MAX_DELETES` deletions, at most :data:`MAX_DELETE_BYTES` bytes
  freed, and a reference-scan budget of :data:`MAX_SCAN_BYTES` transcript bytes
  -- exceeding the budget SKIPS the deletion phase rather than guessing.
- Fail-closed: any session entry that cannot be listed or stat'ed, and any
  transcript that exists but cannot be read, aborts the deletion phase for
  that pass (a reference we cannot see is a reference we must assume).
- Interval: the runner gates passes with a stamp file (``SWEEP_INTERVAL_S``),
  so a machine runs at most one pass per interval no matter how many sessions
  start.

WHY THE REFERENCE SCAN READS RAW BYTES. The digest-reference grammar is sync's
byte regex -- the same scan move/sync uses to decide which blobs travel with a
moved conversation (memo §2.4). It is imported rather than respelled so the two
scanners cannot drift; a journal that references a digest in any other spelling
would already be a sync bug.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Iterable, Mapping

from local_operator.session.attachments import ATTACHMENTS_DIRNAME
from local_operator.supplements.contract import JOURNALED_STATES, SUPPLEMENT_CUSTOM_TYPE

logger = logging.getLogger(__name__)

#: The mime marker of a supplement component blob. ONLY ``SupplementRunner._store``
#: writes this mime into the store; ``cache_media`` refuses non-media kinds and the
#: message externalizer defaults to image/audio mimes, so this filter is what keeps
#: the sweep from ever touching a user artifact.
COMPONENT_MIME: Final = "text/html"

#: Blobs younger than this are never candidates. A job's whole lifetime is bounded
#: by ``supplements.timeoutS`` (90 s default) plus its terminal append; this window
#: is 40x that bound, so an in-flight job's blob cannot be swept even if the pass
#: runs at the worst possible moment. See the module docstring.
SWEEP_GRACE_S: Final = 60 * 60.0

#: How often a machine runs a pass, however many sessions start inside the window.
SWEEP_INTERVAL_S: Final = 6 * 60 * 60.0

#: Per-pass bounds. Examined oldest-first, so a store with more orphans than one
#: pass can take still converges across passes.
MAX_CANDIDATES: Final = 64
MAX_DELETES: Final = 32
MAX_DELETE_BYTES: Final = 4 * 1024 * 1024
#: The reference scan's transcript-byte budget. Exceeding it skips the deletion
#: phase for the pass (the guard against a pathological install scanning forever);
#: a normal install is a small fraction of this.
MAX_SCAN_BYTES: Final = 1 << 30  # 1 GiB

#: Stale-row settlements per pass, and the state/error spelling written. ``disposed``
#: is the memo §7 spelling for a job cut because its runtime went away.
MAX_STALE_ROWS: Final = 8
DISPOSED_ERROR: Final = "disposed"

#: The store names, mirrored from ``session/attachments.py`` (private helpers there;
#: the patterns are re-spelled here as the one layout constant this module needs).
_CONTENT_SUFFIX: Final = ".bin"
_META_SUFFIX: Final = ".json"

_TRANSCRIPT_NAME: Final = "transcript.jsonl"
_SESSIONS_DIRNAME: Final = "sessions"


def _attachment_ref_re():
    """The digest-reference scanner, imported from sync so the two cannot drift.

    Private on purpose: ``network/sync.py`` owns the grammar and this module is
    its second READER, never a second definition. See the module docstring.
    """
    from local_operator.network.sync import _ATTACHMENT_REF

    return _ATTACHMENT_REF


@dataclass(frozen=True)
class BlobCandidate:
    """One unreferenced-so-far component blob, with everything the pass needs."""

    digest: str
    content: Path
    meta: Path
    mtime: float
    size: int


@dataclass
class SweepReport:
    """What one store pass did -- returned for the DEBUG line and the tests."""

    candidates: int = 0
    scanned_sessions: int = 0
    scanned_bytes: int = 0
    deleted: int = 0
    bytes_freed: int = 0
    skipped: str = ""
    elapsed_ms: float = 0.0
    #: Digests a pass declined to reclaim (referenced, or dropped by a cap), for
    #: tests that pin the safety rules by name rather than by "nothing broke".
    kept: list[str] = field(default_factory=list)


def _load_sidecar(path: Path) -> Mapping[str, Any] | None:
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) else None


def blob_candidates(
    attachments_dir: Path,
    *,
    older_than: float,
    limit: int = MAX_CANDIDATES,
) -> list[BlobCandidate]:
    """Component blobs older than ``older_than`` (a timestamp), oldest first.

    A blob is a candidate only when its sidecar says ``text/html`` -- the
    supplement-only mime -- so media artifacts are never examined. A missing or
    damaged sidecar is skipped (the store documents content-without-sidecar as
    merely unused disk, and this module must not guess a mime); a stat failure
    on either half is skipped. Deterministic: sorted by ``(mtime, digest)``.
    """
    candidates: list[BlobCandidate] = []
    try:
        entries = list(attachments_dir.iterdir())
    except OSError:
        return []
    for path in entries:
        name = path.name
        if not name.endswith(_CONTENT_SUFFIX):
            continue
        digest = name[: -len(_CONTENT_SUFFIX)]
        if not digest or any(ch not in "0123456789abcdef" for ch in digest):
            continue
        meta_path = attachments_dir / (digest + _META_SUFFIX)
        try:
            stat = path.stat()
        except OSError:
            continue
        if stat.st_mtime >= older_than:
            continue
        meta = _load_sidecar(meta_path)
        if meta is None or str(meta.get("mime_type", "")) != COMPONENT_MIME:
            continue
        candidates.append(
            BlobCandidate(
                digest=digest,
                content=path,
                meta=meta_path,
                mtime=stat.st_mtime,
                size=stat.st_size,
            )
        )
    candidates.sort(key=lambda candidate: (candidate.mtime, candidate.digest))
    return candidates[:limit]


def _session_transcripts(sessions_dir: Path) -> tuple[list[Path], list[str]]:
    """Every session transcript under ``sessions_dir``, plus discovery problems.

    Discovery is the first half of the fail-closed contract -- an entry this
    scan cannot enumerate or stat may hold a transcript that references a
    candidate -- so "cannot look" must reach the caller as loudly as "cannot
    read": both land in the ``problems`` channel ``referenced_digests`` folds
    into the ``problem`` its callers gate on. Round-2 review R2-1: an
    unlistable ``sessions/`` returned ``[]`` and a stat failure on an entry was
    skipped, both with no problem recorded, so ``store_sweep``'s gate never
    fired and a session dir without ``+x`` (transcript intact and referencing)
    let the pass delete the blob its transcript still named.

    The one absence that is NOT a problem: ``sessions_dir`` itself missing --
    there is no sessions dir, so there is no transcript and no reference to
    miss, the same rule ``referenced_digests`` applies to a transcript deleted
    mid-scan. Returns ``(transcripts, problems)``.
    """
    try:
        children = list(sessions_dir.iterdir())
    except FileNotFoundError:
        return [], []
    except OSError:
        return [], [f"list:{sessions_dir}"]
    found: list[Path] = []
    problems: list[str] = []
    for child in children:
        try:
            child_is_dir = child.is_dir()
        except OSError:
            problems.append(f"stat:{child}")
            continue
        if not child_is_dir:
            continue
        transcript = child / _TRANSCRIPT_NAME
        try:
            transcript_is_file = transcript.is_file()
        except OSError:
            problems.append(f"stat:{transcript}")
            continue
        if transcript_is_file:
            found.append(transcript)
    return found, problems


def referenced_digests(
    sessions_dir: Path,
    *,
    budget_bytes: int = MAX_SCAN_BYTES,
) -> tuple[set[str], int, int, str]:
    """Every digest referenced by ANY session transcript.

    Returns ``(digests, sessions_scanned, bytes_scanned, problem)``. The scan is
    deliberately NOT narrowed by transcript mtime. It once was -- a transcript last
    written before a candidate's mtime was skipped, on the theory that a reference
    is appended no earlier than the blob it names -- and that theory is FALSE for
    this feature's own rows: supplement rows are appended with
    ``preserve_mtime=True``, so the restored mtime is pinned back to the session's
    last real write, which predates the blob (``_store`` runs first), and a
    steer/restart can append a reference arbitrarily long after that write. A
    skipped referencing transcript is exactly how the pass deletes a REFERENCED
    blob -- silent data loss, reproduced in round-1 review R1. Completeness is the
    write-safety contract; the pass stays bounded by the scan budget below, which
    fails closed.

    ``problem`` is non-empty when the scan could NOT be trusted: a session
    entry that could not be listed or stat'ed, a transcript that is not
    readable, or a budget overrun. Callers must treat a non-empty
    ``problem`` as "assume the missing references exist" and delete nothing --
    fail-closed, because an unread reference is exactly the case that must not
    become a deletion. ``FileNotFoundError`` (a session dir deleted mid-scan)
    is not a problem: there is no transcript to read and will not be one.
    """
    regex = _attachment_ref_re()
    digests: set[str] = set()
    sessions = 0
    scanned_bytes = 0
    transcripts, problems = _session_transcripts(sessions_dir)
    for transcript in transcripts:
        try:
            stat = transcript.stat()
        except FileNotFoundError:
            continue
        except OSError:
            problems.append(f"stat:{transcript}")
            continue
        if scanned_bytes + stat.st_size > budget_bytes:
            problems.append("budget")
            break
        try:
            payload = transcript.read_bytes()
        except FileNotFoundError:
            continue
        except OSError:
            problems.append(f"read:{transcript}")
            continue
        sessions += 1
        scanned_bytes += len(payload)
        digests.update(match.decode("ascii") for match in regex.findall(payload))
    problem = ",".join(problems[:4])
    if len(problems) > 4:
        problem += f",+{len(problems) - 4}"
    return digests, sessions, scanned_bytes, problem


def store_sweep(
    config_root: Path,
    *,
    now: float | None = None,
    grace_s: float = SWEEP_GRACE_S,
    max_candidates: int = MAX_CANDIDATES,
    max_deletes: int = MAX_DELETES,
    max_delete_bytes: int = MAX_DELETE_BYTES,
    max_scan_bytes: int = MAX_SCAN_BYTES,
) -> SweepReport:
    """One store pass: examine, verify, reclaim -- under the module's bounds.

    Synchronous on purpose: the runner runs it on a worker thread, and the tests
    call it directly, so the safety rules are exercised without an event loop.
    Never raises: every failure path leaves blobs in place and records why in
    ``SweepReport.skipped``.
    """
    started = time.perf_counter()
    report = SweepReport()
    root = Path(config_root)
    try:
        candidates = blob_candidates(
            root / ATTACHMENTS_DIRNAME,
            older_than=(now or time.time()) - grace_s,
            limit=max_candidates,
        )
        report.candidates = len(candidates)
        if not candidates:
            return report
        # Verify against EVERY session first; only a complete, trusted scan may
        # delete. The scan is deliberately NOT narrowed by transcript mtime --
        # ``referenced_digests``' docstring records why that ordering cannot be
        # relied on (preserve_mtime appends; round-1 review R1). The scan budget
        # bounds the pass instead and fails closed -- as does a discovery
        # failure, which reaches this same ``problem`` gate (round-2 review R2-1).
        referenced, sessions, scanned, problem = referenced_digests(
            root / _SESSIONS_DIRNAME, budget_bytes=max_scan_bytes
        )
        report.scanned_sessions = sessions
        report.scanned_bytes = scanned
        if problem:
            report.skipped = f"scan:{problem}"
            report.kept = [candidate.digest for candidate in candidates]
            return report
        deleted = 0
        freed = 0
        for candidate in candidates:
            if candidate.digest in referenced:
                report.kept.append(candidate.digest)
                continue
            if deleted >= max_deletes or freed + candidate.size > max_delete_bytes:
                report.kept.append(candidate.digest)
                continue
            # Content first, then the sidecar: a crash between the two leaves
            # content-without-sidecar, the store's documented "merely unused
            # disk" state, which the next pass retries -- the reverse order
            # would leave a sidecar resolving to nothing.
            try:
                candidate.content.unlink()
                freed += candidate.size
                deleted += 1
            except FileNotFoundError:
                continue
            except OSError:
                report.kept.append(candidate.digest)
                continue
            try:
                candidate.meta.unlink()
            except OSError:
                pass  # content gone; a stale sidecar reads as missing (get's contract)
        report.deleted = deleted
        report.bytes_freed = freed
        return report
    except Exception:  # noqa: BLE001 -- the sweep fails open: worst case, bytes stay
        logger.debug("supplement store sweep failed", exc_info=True)
        report.skipped = report.skipped or "error"
        return report
    finally:
        report.elapsed_ms = (time.perf_counter() - started) * 1000.0


def stale_rows(entries: Iterable[Any]) -> list[dict[str, Any]]:
    """Newest row per anchor that is still NON-terminal -- the stale candidates.

    Applied to a journal this runtime OWNS: at runner-build time no job of this
    process has run, so a non-terminal supplement row is a leftover from a
    runtime that died (a reaper cut, a crash, a kill). Pure: reads the entry
    shape ``transcript.entries()`` yields and nothing else; the caller decides
    (under the write lock) whether each is still worth settling.
    """
    newest: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if getattr(entry, "type", "") != "custom":
            continue
        payload = entry.payload if isinstance(entry.payload, dict) else {}
        details = payload.get("details")
        if not isinstance(details, dict):
            continue
        if payload.get("custom_type") != SUPPLEMENT_CUSTOM_TYPE:
            continue
        anchor, version = details.get("anchor"), details.get("version")
        if not isinstance(anchor, str) or not isinstance(version, int) or isinstance(version, bool):
            continue
        held = newest.get(anchor)
        if held is None or version >= int(held.get("version", 0) or 0):
            newest[anchor] = details
    terminal = frozenset(JOURNALED_STATES - {"decided", "queued"})
    return [details for details in newest.values() if str(details.get("state", "")) not in terminal]


def sweep_stamp_path(config_root: Path) -> Path:
    """The interval gate's file. One per config root -- one machine, one clock."""
    return Path(config_root) / ".supplements-sweep.stamp"


def sweep_due(config_root: Path, *, now: float | None = None) -> bool:
    """Whether a pass is due, and if so, CLAIMS the interval by writing the stamp.

    The claim is the write (``O_CREAT`` + replace): two sessions starting at
    once may both see "due" and both pass -- the second finds no candidates or
    no work, which is a wasted pass, never a wrong one -- but the stamp keeps
    every later session inside the interval out. A crash mid-pass therefore
    delays the next attempt one interval, which is the safe direction for a
    cleanup pass. Never raises: an unwritable stamp means "not due" (the sweep
    is optional; a broken gate must not run it unboundedly).
    """
    stamp = sweep_stamp_path(config_root)
    moment = time.time() if now is None else now
    try:
        last = float(stamp.read_text(encoding="utf-8").strip() or 0.0)
    except FileNotFoundError:
        last = 0.0
    except (OSError, ValueError):
        return False
    if moment - last < SWEEP_INTERVAL_S:
        return False
    try:
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(str(moment), encoding="utf-8")
    except OSError:
        return False
    return True
