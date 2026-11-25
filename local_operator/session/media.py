"""Retained session media, addressed by transcript identity rather than paths.

A view grant to one conversation is not a grant to the whole attachment store.
Resolve the published entry and family-specific index first, then its retained
content. Origin paths and source URLs are display metadata, never byte sources.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import logging
import re
import threading
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionMedia:
    mime: str
    media_id: str
    total: int
    path: Path | None = None
    data: bytes | None = None
    fingerprint: tuple[int, int, int] | None = None

    def __post_init__(self) -> None:
        # Stored/imported MIME is descriptive, not HTTP-header authority. The
        # same resolver serves local files and remote chunks, so enforce this
        # invariant here rather than only on the mesh receiver.
        if not self.mime.isascii() or "\r" in self.mime or "\n" in self.mime:
            object.__setattr__(self, "mime", "application/octet-stream")

    def read(self, offset: int, length: int) -> bytes:
        if self.path is not None:
            import os

            with self.path.open("rb") as handle:
                before = os.fstat(handle.fileno())
                if (before.st_ino, before.st_size, before.st_mtime_ns) != self.fingerprint:
                    raise OSError("retained media changed")
                handle.seek(offset)
                data = handle.read(length)
                after = os.fstat(handle.fileno())
                if (after.st_ino, after.st_size, after.st_mtime_ns) != self.fingerprint:
                    raise OSError("retained media changed")
                return data
        return (self.data or b"")[offset : offset + length]


def _entry(directory: Path, entry_id: str) -> Any:
    from local_operator.session.transcript import read_entry_by_id, read_transcript_page

    entry = read_entry_by_id(directory, entry_id)
    if entry is not None:
        return entry
    # Match both phone tool-row spellings. Locate the assistant prefix rather
    # than splitting once: call ids themselves may contain colons, and the
    # published composite must still name its actual parent call.
    call_id = entry_id[3:] if entry_id.startswith("tc-") else ""
    if not call_id:
        for cut in reversed([position for position, char in enumerate(entry_id) if char == ":"]):
            parent = read_entry_by_id(directory, entry_id[:cut])
            candidate_id = entry_id[cut + 1 :]
            calls = (parent.payload.get("tool_calls") or []) if parent is not None else []
            if not isinstance(calls, list):
                continue
            if any(isinstance(call, dict) and call.get("id") == candidate_id for call in calls):
                call_id = candidate_id
                break
    if not call_id:
        return None
    before = None
    seen: set[str] = set()
    while True:
        if before is not None:
            if before in seen:
                return None  # a concurrently replaced journal reconciled us
            seen.add(before)
        page = read_transcript_page(directory, before_id=before)
        for candidate in reversed(page.entries):
            if (
                candidate.payload.get("role") == "tool"
                and candidate.payload.get("tool_call_id") == call_id
            ):
                return candidate
        if not page.has_more or not page.entries:
            return None
        before = page.entries[0].id


def session_media(
    root: Path, session_id: str, entry_id: str, index: int, kind: str
) -> SessionMedia | None:
    """Resolve image/video/file bytes retained by this exact user conversation.

    Store-backed videos are hashed and read in existing mesh-sized chunks; they
    are never materialized as a whole-video bytes/base64 object. Hash validation
    preserves AttachmentStore's bit-rot contract and makes a changed object fail
    the receiver's expected-media identity on subsequent chunks.
    """
    from local_operator.network.sync import SYNC_CHUNK_BYTES
    from local_operator.resume import is_user_session
    from local_operator.session.attachments import store_for_transcript_dir
    from local_operator.session.compression import has_transcript

    if session_id in ("", ".", "..") or Path(session_id).name != session_id:
        return None
    if type(index) is not int or index < 0 or kind not in ("image", "video", "file"):
        return None
    directory = root / "sessions" / session_id
    if not has_transcript(directory) or not is_user_session(directory):
        return None
    entry = _entry(directory, entry_id)
    if entry is None or entry.type != "message":
        return None
    # A retained pre-compaction row remains addressable, but an explicitly
    # pruned row does not. Scan only prune records, without hydrating media.
    from local_operator.session.compression import transcript_for_read
    from local_operator.session.transcript import TranscriptEntry

    journal = transcript_for_read(directory, read_only=True)
    if journal is None:
        return None
    try:
        with journal.open("rb") as handle:
            for raw in handle:
                if b'"prune"' not in raw:
                    continue
                marker = TranscriptEntry.from_json(raw.decode("utf-8", errors="replace"))
                if (
                    marker is not None
                    and marker.type == "prune"
                    and marker.payload.get("target") == entry.id
                ):
                    return None
    except OSError:
        return None
    content = entry.payload.get("content")
    if not isinstance(content, list):
        return None
    blocks = []
    for block in content:
        if not isinstance(block, dict):
            continue
        # Same family index as mobile projection._image_refs/_video_refs.
        artifact_kind = block.get("kind")
        block_type = block.get("type")
        family = None
        if artifact_kind in ("image", "video", "audio") and any(
            field in block for field in ("content_type", "attachment", "source_url")
        ):
            family = artifact_kind
        elif block_type == "document" or (
            not block_type and "filename" in block and "mime_type" in block
        ):
            family = "file"
        elif block_type == "image" or (
            not block_type
            and not artifact_kind
            and any(field in block for field in ("data", "attachment"))
            and not str(block.get("mime_type", "")).startswith("audio/")
        ):
            # Default ImageContent fields are omitted in durable rows: an
            # inline PNG can be exactly {data: ...}, not {type: image, ...}.
            family = "image"
        if family == kind:
            blocks.append(block)
    if index >= len(blocks):
        return None
    block = blocks[index]
    mime = str(
        block.get("content_type")
        or block.get("mime_type")
        or {"image": "image/png", "video": "video/mp4", "file": "application/octet-stream"}[kind]
    )
    digest = block.get("attachment")
    if digest:
        # The digest came from the retained row, not the phone. Validate it
        # anyway: a damaged journal must not turn a store lookup into traversal.
        if not isinstance(digest, str) or re.fullmatch(r"[a-f0-9]{32}", digest) is None:
            return None
        store = store_for_transcript_dir(directory)
        retained_mime = store.mime_type_for(digest)
        if retained_mime is None:
            return None
        mime = str(block.get("content_type") or block.get("mime_type") or retained_mime)
        path = store.path_for(digest)
        try:
            before = path.stat()
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                while chunk := handle.read(SYNC_CHUNK_BYTES):
                    hasher.update(chunk)
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                return None
            identity = hasher.hexdigest()
            if not identity.startswith(digest):
                return None
            return SessionMedia(
                mime,
                identity,
                after.st_size,
                path=path,
                fingerprint=(after.st_ino, after.st_size, after.st_mtime_ns),
            )
        except OSError:
            return None
    raw = block.get("data")
    if not isinstance(raw, str):
        return None
    try:
        data = base64.b64decode(raw)
    except (ValueError, binascii.Error):
        return None
    return SessionMedia(mime, hashlib.sha256(data).hexdigest(), len(data), data=data)


# ---------------------------------------------------------------------------
# Turn supplements (Highlights): the ONE byte policy for featured files, and the
# digest-addressed component/sound sources. Shared by the relay's local routes
# (``mobile/daemon.py``) and the mesh owner (``mobile/remote_sessions.owner_read``)
# so "what a featured file may be" has a single definition on every machine:
# a remote viewer is answered by the owner running exactly the function the
# owner's own phone route runs, never by a re-implementation that could drift.
# ---------------------------------------------------------------------------

#: A supplement JOB id: ``persistence.new_job_id`` mints 12 lowercase hex, and the
#: file route keys on it. Shape-checked before any lookup so a crafted job segment
#: can never be anything but a miss (it is a lookup key, never a path).
SUPPLEMENT_JOB = re.compile(r"^[a-f0-9]{12}$")

#: The file route's per-request byte cap (memo §2.7: "a 10 MB cap"), re-applied here
#: rather than trusted from the decision, because the bytes on disk are what is read.
SUPPLEMENT_FILE_MAX_BYTES = 10 * 1024 * 1024

#: The SUFFIX -> Content-Type allowlist for the file route. Never ``text/html``,
#: ``image/svg+xml``, ``application/xhtml+xml``, or any ``+xml`` (memo §2.7, S15):
#: the served type is decided by this table alone, never by the bytes, so a ``.txt``
#: whose body opens with ``<script>`` is still served as ``text/plain`` under
#: ``nosniff`` and cannot execute. ``svg`` and ``xml`` are deliberately absent.
#: A remote viewer re-checks a peer's reply against this table's VALUES too, so a
#: buggy or hostile owner cannot get ``text/html`` served from the viewer's origin.
SUPPLEMENT_FILE_TYPES: dict[str, str] = {
    ".txt": "text/plain",
    ".log": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".json": "application/json",
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".avif": "image/avif",
    # Video for the written-video preview tile. Inert media types (the player decodes;
    # nothing executes), still under the 10 MB cap and ``nosniff`` like every entry.
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
}

#: A stored component's raw size cap on the REMOTE document read. The viewer
#: assembles the document itself (so the CSP comes from the build serving the
#: phone), which means it must buffer the raw blob; 1 MiB is ~10x the largest
#: referenced component measured across 1,500 local sessions (67 blobs, median
#: 25.7 KB, max 97 KB — architect note for the #4057 follow-on).
SUPPLEMENT_DOCUMENT_MAX_BYTES = 1024 * 1024

#: Refusal tokens of :func:`resolve_supplement_file`. Tokens, not exceptions: the
#: local route maps them to HTTP statuses and the mesh owner to refusal codes, and
#: both must stay exhaustive over the same small set.
SUPPLEMENT_MISSING = "missing"
SUPPLEMENT_NO_CWD = "no-cwd"
SUPPLEMENT_TOO_LARGE = "too-large"

#: WHO MAY READ A FEATURED FILE THROUGH A MESH VIEW GRANT: exactly what the owner's
#: own phone may (operator decision, 2026-11-15: "Every file the Highlight names, with
#: the same rules as on that computer"). A remote viewer is answered by the OWNER
#: running this module's :func:`resolve_supplement_file` with the owner's OWN live
#: cwd — cwd, ``~`` and the session's scratchpad as roots, the same §4.3 denylist,
#: regular file, 10 MB cap, suffix allowlist. The reasoning: view grants go to the
#: operator's own devices, so a Highlight's file opening on the phone is the same
#: disclosure as opening it on the computer. This deliberately goes beyond
#: ``network/files.py``'s pull scope (pad + referenced digests); docs/mobile.md
#: records the decision. There is no second remote policy: narrowing it again means
#: adding one here, beside this note.


def resolve_supplement_file(
    directory: Path,
    job: str,
    index: int,
    *,
    cwd: str,
    fallback_cwd: str = "",
) -> SessionMedia | str:
    """``row["files"][index]`` of supplement ``job`` as bytes, or a refusal token.

    Extracted verbatim from the relay's local file route (memo §2.7) so the mesh
    owner applies the identical policy: the newest version of the job's row names
    the file; the resolved path must stay under ``cwd``, ``~`` or the session's own
    scratchpad (a tampered row does not become a read primitive); the §4.3
    sensitive-path denylist is re-applied; the file must be a regular file of at
    most :data:`SUPPLEMENT_FILE_MAX_BYTES`; the type comes from
    :data:`SUPPLEMENT_FILE_TYPES` alone.

    ``cwd`` is the caller's LIVE runtime record's cwd or ``""``. An empty cwd makes
    a cwd-relative path ``no-cwd`` rather than resolved against a guess — the class
    of silent wrong-directory this codebase refuses. Locally that comes from the
    daemon's session table; on the mesh owner, from its OWN registry record, never
    from the viewer (nothing the viewer sends is a path).

    ``fallback_cwd`` is the DURABLE record's cwd for a session with no live runtime
    (an ended session's frontend checkpoint, or its marker/wake index). It is a
    recollection, not a vouched root, so it only JOINS a cwd-relative path and only
    when the resolved file still matches the row's recorded ``size_bytes`` and
    ``mtime`` exactly (a row without that fingerprint stays ``no-cwd``): a file that
    changed since the Highlight was pinned, or a directory that is no longer the
    one the agent worked in, is refused rather than served. It is never an allowed
    root and never the denylist's cwd, so a recorded cwd such as ``/`` cannot widen
    reads beyond ``~`` and the session's scratchpad.

    The SAME function answers a mesh viewer on the owner (see the scope note above
    this function): there is no remote-only branch to drift.

    The returned :class:`SessionMedia` holds the bytes (<= 10 MB, read once), their
    sha256 as ``media_id`` and the stat fingerprint: a mesh reader re-resolves per
    chunk and a file that changed between chunks fails ``expected_media_id``.
    """
    import os

    from local_operator.supplements import denylist
    from local_operator.supplements.contract import SUPPLEMENT_CUSTOM_TYPE

    if not SUPPLEMENT_JOB.fullmatch(job):
        return SUPPLEMENT_MISSING
    try:
        from local_operator.session.transcript import read_custom_entries

        # A byte-level read of the one custom type, oldest first (the fold below
        # is order-sensitive on version ties), instead of constructing a whole
        # writer ``Transcript`` per file request.
        entries = list(reversed(read_custom_entries(directory, SUPPLEMENT_CUSTOM_TYPE)))
    except Exception:  # noqa: BLE001 -- an odd transcript serves no file
        return SUPPLEMENT_MISSING
    newest: dict[str, Any] | None = None
    for entry in entries:
        if entry.type != "custom":
            continue
        payload = entry.payload if isinstance(entry.payload, dict) else {}
        details = payload.get("details")
        if not isinstance(details, dict) or details.get("job") != job:
            continue
        version = details.get("version")
        if not isinstance(version, int) or isinstance(version, bool):
            continue
        if newest is None or version >= int(newest.get("version", -1)):
            newest = details
    if newest is None:
        return SUPPLEMENT_MISSING
    files = newest.get("files")
    if not isinstance(files, list) or not 0 <= index < len(files):
        return SUPPLEMENT_MISSING
    recorded = files[index] if isinstance(files[index], dict) else {}
    raw = recorded.get("path")
    if not isinstance(raw, str) or not raw:
        return SUPPLEMENT_MISSING

    home = str(Path.home())
    expanded = os.path.expanduser(raw)
    must_match: tuple[int, float] | None = None
    if not os.path.isabs(expanded):
        if cwd:
            expanded = os.path.join(cwd, expanded)
        else:
            # No live record: a cwd-relative path has no root we can vouch for, and
            # resolving it against a guess is the class of silent wrong-directory
            # this codebase refuses. The durable cwd may stand in only when the
            # file it names is provably the one the row recorded (see the
            # docstring), so a wrong directory cannot serve a look-alike.
            size, mtime = recorded.get("size_bytes"), recorded.get("mtime")
            if (
                not fallback_cwd
                or isinstance(size, bool)
                or not isinstance(size, int)
                or isinstance(mtime, bool)
                or not isinstance(mtime, (int, float))
            ):
                return SUPPLEMENT_NO_CWD
            must_match = (size, float(mtime))
            expanded = os.path.join(fallback_cwd, expanded)
    resolved_path = Path(os.path.realpath(expanded))
    # The row's own session scratchpad is a root beside cwd/home (a relocated config
    # dir may sit outside ``~``), mirroring the prefilter that admitted the path.
    from local_operator.scratchpad import scratchpad_root

    own_pad = scratchpad_root(directory)
    allowed_roots = [
        Path(os.path.realpath(root))
        for root in (cwd, home, str(own_pad) if own_pad is not None else "")
        if root
    ]
    if not any(
        resolved_path == root or resolved_path.is_relative_to(root) for root in allowed_roots
    ):
        return SUPPLEMENT_MISSING
    # The row's own session scratchpad is servable (the prefilter admitted it under
    # the same carve-out); every credential rule and every OTHER session's store
    # stay refused. Derived from the requested session's directory, never the env.
    if denylist.is_sensitive_for_supplements(
        str(resolved_path),
        cwd=cwd or home,
        session_scratchpad=str(own_pad) if own_pad is not None else None,
    ):
        _logger.info("supplement file refused by the denylist for %s", directory.name)
        return SUPPLEMENT_MISSING
    try:
        if not resolved_path.is_file():
            return SUPPLEMENT_MISSING
        # The pre-read stat is the fingerprint: a second, post-read stat would turn
        # bytes already read into a 404 if the file were unlinked in between (the
        # local route never did that), and the bytes-backed media never re-checks it.
        before = resolved_path.stat()
        if must_match is not None and (before.st_size, before.st_mtime) != must_match:
            return SUPPLEMENT_NO_CWD
        if before.st_size > SUPPLEMENT_FILE_MAX_BYTES:
            return SUPPLEMENT_TOO_LARGE
        fingerprint = (before.st_ino, before.st_size, before.st_mtime_ns)
        # Every policy check above has already run for THIS request; the memo only
        # skips re-reading and re-hashing bytes whose stat identity is unchanged.
        memo_key = (str(directory), job, index, str(resolved_path), fingerprint)
        cached = _resolved_file_memo_get(memo_key)
        if cached is not None:
            return cached
        data = resolved_path.read_bytes()
    except OSError:
        return SUPPLEMENT_MISSING
    if len(data) > SUPPLEMENT_FILE_MAX_BYTES:
        return SUPPLEMENT_TOO_LARGE
    mime = SUPPLEMENT_FILE_TYPES.get(resolved_path.suffix.lower())
    if mime is None:
        return SUPPLEMENT_MISSING
    media = SessionMedia(
        mime,
        hashlib.sha256(data).hexdigest(),
        len(data),
        data=data,
        fingerprint=fingerprint,
    )
    _resolved_file_memo_put(memo_key, media)
    return media


#: Owner-side memo for :func:`resolve_supplement_file` (security review S-3). A mesh
#: viewer re-resolves per 512 KiB chunk, so without it one 10 MB download costs ~20
#: full reads + sha256s, and a ``length=1`` viewer can scale that per byte. Keyed by
#: the session directory, job, index, resolved path AND the pre-read stat identity
#: ``(st_ino, st_size, st_mtime_ns)``: a rewrite changes the stat, misses the memo,
#: re-hashes, and the new sha256 fails ``expected_media_id`` (``remote_media_changed``)
#: exactly as before. Bounded to a few entries (each <= 10 MB) so the owner never
#: holds more than ~40 MB; a lock because resolvers run on worker threads.
_RESOLVED_FILE_MEMO_ENTRIES = 4
_resolved_file_memo: OrderedDict[tuple[Any, ...], SessionMedia] = OrderedDict()
_resolved_file_memo_lock = threading.Lock()


def _resolved_file_memo_get(key: tuple[Any, ...]) -> SessionMedia | None:
    with _resolved_file_memo_lock:
        media = _resolved_file_memo.get(key)
        if media is not None:
            _resolved_file_memo.move_to_end(key)
        return media


def _resolved_file_memo_put(key: tuple[Any, ...], media: SessionMedia) -> None:
    with _resolved_file_memo_lock:
        _resolved_file_memo[key] = media
        _resolved_file_memo.move_to_end(key)
        while len(_resolved_file_memo) > _RESOLVED_FILE_MEMO_ENTRIES:
            _resolved_file_memo.popitem(last=False)


def journal_references_component(payload: bytes, digest: str) -> bool:
    """Whether a ``supplement_v1`` row of this journal references ``digest``.

    §2.7's letter for the document route: the reference is read per
    ``supplement_v1`` row. Deliberately narrower than a raw session-wide scan
    with sync's byte grammar -- the grammar also matches ``"attachment"``
    references on other row types (message media carries the same key today),
    and a future text-mime producer must not become servable through this door
    without the authorisation being reconsidered (round-1 review R4). The cheap
    pre-filter keeps JSON parsing to the lines that can be supplement rows.

    Module level (not a relay closure) so the mesh owner authorises a remote
    document read by the same rule as the owner's own phone route.
    """
    import json

    from local_operator.network.sync import _ATTACHMENT_REF
    from local_operator.supplements.contract import SUPPLEMENT_CUSTOM_TYPE

    marker = SUPPLEMENT_CUSTOM_TYPE.encode("ascii")
    needle = digest.encode("ascii")
    for line in payload.splitlines():
        if marker not in line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict):
            continue
        row_payload = row.get("payload")
        if not isinstance(row_payload, dict):
            continue
        if row_payload.get("custom_type") != SUPPLEMENT_CUSTOM_TYPE:
            continue
        if any(match == needle for match in _ATTACHMENT_REF.findall(line)):
            return True
    return False


def journal_references_sound(lines: Iterable[str], digest: str) -> bool:
    """Whether a ``supplement_v1`` row's ``sounds[]`` names ``digest``.

    THE SOUND ROUTES' THIRD AUTHORISATION FACT, and deliberately a PRECISE
    reader rather than :func:`journal_references_component`: that helper
    matches any ``attachment`` reference on a supplement row, which includes
    the COMPONENT document digests. Serving an audio route off a component
    digest would let the bytes family be reached through a reference that
    never meant it — the round-1 review's own distinction (B1's sibling).

    Lines, not bytes: after #2798 a settled transcript may be archived, so
    every caller passes ``session.compression.iter_lines(directory)``. A
    plain ``transcript.jsonl`` probe would read a compressed session as
    "no reference" and 404 its own sounds — the guard
    ``test_no_plain_name_probe_outside_the_resolver`` exists for exactly
    this.

    Read per row, with the same cheap pre-filter (the marker and the digest
    must both appear on the line before a parse is worth doing). A beat that
    has not yet been journaled references nothing yet; the live projection
    only paints a settled beat, whose row the job writes at settle.

    Module level (not a relay closure) so the mesh owner authorises a remote
    sound read by the same rule as the owner's own phone route.
    """
    import json

    from local_operator.supplements.contract import SUPPLEMENT_CUSTOM_TYPE

    marker = SUPPLEMENT_CUSTOM_TYPE
    for line in lines:
        if marker not in line or digest not in line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict):
            continue
        row_payload = row.get("payload")
        if not isinstance(row_payload, dict):
            continue
        if row_payload.get("custom_type") != SUPPLEMENT_CUSTOM_TYPE:
            continue
        details = row_payload.get("details")
        if not isinstance(details, dict):
            continue
        sounds = details.get("sounds")
        if not isinstance(sounds, list):
            continue
        for sound in sounds:
            if isinstance(sound, dict) and sound.get("attachment") == digest:
                return True
    return False


def supplement_digest_media(
    root: Path, session_id: str, digest: str, kind: str
) -> SessionMedia | str:
    """A supplement component (``document``) or sound (``sound``) blob, or a token.

    The mesh owner's source for the two digest-addressed supplement families. The
    authorisation is the local routes' own: the session is a user conversation on
    THIS device, and a ``supplement_v1`` row of its journal references the digest
    in the family's own slot (any attachment reference for a component, ``sounds[]``
    for a sound — a component digest never reaches the audio family). Referenced
    digests are already inside the view-grant policy (``network/files.py``: "digests
    its transcript references"), so serving them remotely widens nothing.

    The returned :class:`SessionMedia` is path-backed with a sha256 identity that
    must extend the store digest (AttachmentStore's bit-rot contract), so a blob
    replaced between chunks fails ``expected_media_id`` at the viewer.
    """
    from local_operator.network.sync import SYNC_CHUNK_BYTES
    from local_operator.resume import is_user_session
    from local_operator.session import compression
    from local_operator.session.attachments import store_for_transcript_dir
    from local_operator.supplements.contract import SUPPLEMENT_CUSTOM_TYPE, is_digest

    if session_id in ("", ".", "..") or Path(session_id).name != session_id:
        return SUPPLEMENT_MISSING
    if kind not in ("document", "sound") or not is_digest(digest):
        return SUPPLEMENT_MISSING
    directory = root / "sessions" / session_id
    if not compression.has_transcript(directory) or not is_user_session(directory):
        return SUPPLEMENT_MISSING
    try:
        if kind == "document":
            payload = b"\n".join(
                line.encode("utf-8", "replace")
                for line in compression.iter_lines(directory)
                if SUPPLEMENT_CUSTOM_TYPE in line
            )
            referenced = journal_references_component(payload, digest)
        else:
            referenced = journal_references_sound(compression.iter_lines(directory), digest)
    except (OSError, compression.ArchiveError):
        return SUPPLEMENT_MISSING
    if not referenced:
        return SUPPLEMENT_MISSING
    store = store_for_transcript_dir(directory)
    mime = store.mime_type_for(digest)
    if mime is None:
        return SUPPLEMENT_MISSING
    if kind == "sound":
        # The local sound route's rule: the store's "no stated type" sentinel
        # stands in as mpeg, and anything not audio is NEVER served through the
        # audio family (a component's text/html blob shares the store).
        if mime == "application/octet-stream":
            mime = "audio/mpeg"
        if not mime.startswith("audio/"):
            return SUPPLEMENT_MISSING
    path = store.path_for(digest)
    try:
        before = path.stat()
        if kind == "document" and before.st_size > SUPPLEMENT_DOCUMENT_MAX_BYTES:
            return SUPPLEMENT_TOO_LARGE
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(SYNC_CHUNK_BYTES):
                hasher.update(chunk)
        after = path.stat()
    except OSError:
        return SUPPLEMENT_MISSING
    fingerprint = (after.st_ino, after.st_size, after.st_mtime_ns)
    if (before.st_ino, before.st_size, before.st_mtime_ns) != fingerprint:
        return SUPPLEMENT_MISSING
    identity = hasher.hexdigest()
    if not identity.startswith(digest):
        return SUPPLEMENT_MISSING
    return SessionMedia(mime, identity, after.st_size, path=path, fingerprint=fingerprint)
