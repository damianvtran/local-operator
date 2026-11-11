"""Mesh file pull: one file out of a peer's session, byte for byte (S1).

docs/design/mesh-file-transfer.md, slice S1. WHY THIS EXISTS: a session on a peer
had no byte path back to the head node. ``sessions op='peek'`` returns clipped text
rows, and the only other route was base64 through the model, which corrupted the
PNGs and logs it carried (field report, 2026-11). This module is that byte path,
for READS only (the put half is S2).

THE SHAPE, and why each piece is what it is:

* **Two halves, one link.** ``net_file_get`` is the OWNER'S peer op (``stat``, then
  ``fetch`` chunks, plus ``verify`` for a resume); ``session_file_pull`` is the
  REQUESTER'S local op, run inside THIS device's relay because the relay is the only
  process holding links (``network/cli.py``'s rule). The control socket between the
  CLI and the relay carries PATHS only, never bytes: the relay writes inside its own
  root and the CLI copies the verified result out.
* **Digest first, end to end.** ``stat`` announces the full sha256 before any byte
  moves. Every chunk carries its own digest (``sync.serve_fetch``'s 512 KiB framing,
  well under ``wire.MAX_RECORD_BYTES``), and NOTHING becomes visible — not the
  attachment store, not ``--out`` — until the whole staged file hashes to the
  announced value. A mismatch deletes the part, retries once from zero, then refuses.
* **Resumable.** The requester stages ``<root>/network/staging/files/<sha256>.part``,
  OUTSIDE ``sessions/`` for the same reason ``sync.staging_dir`` is (a half-copied
  file there would read as a session's own content). A re-run of the same file
  resumes from the part's length, after the owner confirms the part is its own
  prefix (``verify``) — never by trusting bytes nobody verified as a whole.
* **Confined at the source.** The owner serves exactly two kinds of thing for a
  session it owns: a ``scratchpad://`` path (parsed by the scratchpad's own parser,
  then the credential vocabulary, then a no-follow regular-file open) or an
  attachment digest THAT SESSION'S TRANSCRIPT REFERENCES. Not the cwd, not the
  config dir, not another session's pad, not the shared store by probing.
* **Authority is the existing table.** ``view`` (``types.OP_CAPABILITY``), checked
  against the requester's member row by the chokepoint, plus ``_session_scope``
  ownership. ``net_sync`` already gives a view member these same bytes, so this
  widens nothing.
* **An old peer is named, not guessed around.** The puller checks
  ``wire.MESH_FILE_V1`` before its first request and answers "update <device>".
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import mimetypes
import os
import re
import stat
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from urllib.parse import unquote, urlsplit

from local_operator.network import sync, wire
from local_operator.network.types import MeshRefusal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.network.relay import PeerLink, RelayServer

logger = logging.getLogger(__name__)

#: One served chunk: the session copy's own framing (``sync.SYNC_CHUNK_BYTES``),
#: so a chunk is one record of ~683 KiB as base64, interleaved with other ops on
#: the link rather than one record that holds the reader for a whole file.
FILE_CHUNK_BYTES = sync.SYNC_CHUNK_BYTES

#: The largest file one pull carries: the scratchpad's own total budget
#: (``scratchpad.SCRATCHPAD_TOTAL_BUDGET_BYTES``), so anything a session could have
#: written to its pad fits, and nothing larger can be asked for by naming it.
FILE_MAX_BYTES = 256 * 1024 * 1024

#: The owner-side deadline of one ``net_file_get`` request. The sync op's own: one
#: request is one stat (a streamed hash of at most ``FILE_MAX_BYTES``) or one chunk.
FILE_OP_DEADLINE_S = sync.SYNC_OP_DEADLINE_S

#: How long the CLI waits for its own relay to finish a whole pull. Generous on
#: purpose: 256 MiB over a slow link is minutes, and a CLI that gave up first would
#: report nothing about a transfer that is still running. A give-up is not a loss —
#: the part stays staged and the next pull resumes it.
FILE_PULL_CLIENT_TIMEOUT_S = 30 * 60.0

#: A staged part nobody resumed for this long is swept at the next pull. Parts are
#: named by digest, so a stale one is only ever disk, never wrong bytes.
PART_MAX_AGE_S = 24 * 3600.0

#: Where the requester stages, relative to its root. Outside ``sessions/``.
STAGING_PARTS: tuple[str, ...] = ("network", "staging", "files")

_ATTACHMENT = re.compile(r"[0-9a-f]{32}")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")

#: Extensions ``mimetypes`` does not know everywhere but a session pad holds often.
#: The store's sidecar carries the type, and ``application/octet-stream`` for a log
#: would make a text file read as a download.
_MIME_FALLBACK = {".log": "text/plain", ".jsonl": "application/x-ndjson", ".md": "text/markdown"}

#: One pull per (root, digest) at a time: two concurrent pulls of the same file
#: would share one ``<sha256>.part`` and interleave writes into it.
_PART_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_PART_LOCKS_GUARD = threading.Lock()


def _refuse(code: str, message: str) -> MeshRefusal:
    return MeshRefusal(code, message)


def _mime_for(name: str) -> str:
    suffix = Path(name).suffix.lower()
    guessed, _encoding = mimetypes.guess_type(name, strict=False)
    return guessed or _MIME_FALLBACK.get(suffix) or "application/octet-stream"


# ---------------------------------------------------------------------------
# The owner's half: what may be served, and serving it
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Source:
    """One servable file: its physical path, the scope it is opened under, and labels."""

    path: Path
    scope: sync.WriteScope
    name: str
    mime: str
    #: The store digest the requester named, or ``""`` for a scratchpad path.
    attachment: str


def _resolve_source(root: Path, session_id: str, frame: dict[str, Any]) -> _Source:
    """The one file ``frame`` names inside ``session_id``, or a refusal naming the rule.

    Called on EVERY phase, not once per pull: each request is authorised on its own
    by the chokepoint, and a file the session stopped referencing (or a pad entry
    that became a link) between two chunks must stop being served at the next one.
    """
    from local_operator.scratchpad import (
        ScratchpadPathError,
        parse_scratchpad_url,
        scratchpad_root,
    )
    from local_operator.session.attachments import ATTACHMENTS_DIRNAME
    from local_operator.supplements.denylist import is_sensitive_for_supplements

    path_ref = str(frame.get("path") or "")
    attachment = str(frame.get("attachment") or "")
    if bool(path_ref) == bool(attachment):
        raise _refuse(
            "bad_request",
            "a file pull names exactly one source: a scratchpad:// path or an attachment digest",
        )
    directory = sync.session_dir(root, session_id)
    if not directory.is_dir():
        raise _refuse("session_not_found", f"session {session_id} is not on this device")
    if attachment:
        if not _ATTACHMENT.fullmatch(attachment):
            raise _refuse(
                "bad_request", f"{attachment!r} is not an attachment digest (32 hex characters)"
            )
        # REFERENCED, NOT MERELY PRESENT (the sound route's rule): the store is shared
        # across every conversation on this device, so serving any digest it holds
        # would let a view member of ONE session probe the bytes of all of them.
        if attachment not in sync.referenced_attachments_in(directory):
            raise _refuse(
                "not_referenced",
                f"attachment {attachment} is not referenced by session {session_id} on this "
                "device, so it is not served",
            )
        store_root = Path(root) / ATTACHMENTS_DIRNAME
        mime = "application/octet-stream"
        try:
            meta = json.loads((store_root / f"{attachment}.json").read_text(encoding="utf-8"))
            if isinstance(meta, dict) and isinstance(meta.get("mime_type"), str):
                mime = meta["mime_type"] or mime
        except (OSError, ValueError):
            pass
        extension = mimetypes.guess_extension(mime) or ".bin"
        return _Source(
            path=store_root / f"{attachment}.bin",
            scope=sync.write_scope(Path(root), store_root),
            name=f"{attachment}{extension}",
            mime=mime,
            attachment=attachment,
        )
    pad = scratchpad_root(directory)
    if pad is None or not pad.is_dir():
        raise _refuse("missing", f"session {session_id} has no scratchpad on this device")
    try:
        # The scratchpad's OWN parser: it already refuses absolute paths, ``..``,
        # dotfiles, nested schemes and a symlink that resolves outside the pad. A
        # second parser here would be a second definition of "inside the pad".
        target = parse_scratchpad_url(path_ref, pad)
    except ScratchpadPathError as error:
        raise _refuse("outside_scratchpad", str(error)) from None
    if target.directory:
        raise _refuse("not_a_file", f"{path_ref} names a directory; pull one file at a time")
    written = pad.joinpath(
        *[
            part
            for part in unquote(urlsplit(path_ref).netloc + urlsplit(path_ref).path).split("/")
            if part
        ]
    )
    # BOTH FORMS, because a pad entry can be a link inside the pad: ``notes.txt`` that
    # resolves to ``id_rsa`` is judged by the name it resolves to, and ``id_rsa`` that
    # resolves to a harmless file by the name it was asked as. The carve-out lifts only
    # the config-dir root rule, and only for THIS session's pad; the credential
    # vocabulary (``.env``, ``id_rsa``, ``*token*``…) applies inside it as everywhere.
    for form in (written, target.path):
        reason = is_sensitive_for_supplements(form, session_scratchpad=str(pad))
        if reason:
            raise _refuse("sensitive", f"{path_ref} is not served: {reason}")
    return _Source(
        path=target.path,
        scope=sync.write_scope(Path(root), pad),
        name=target.path.name,
        mime=_mime_for(target.path.name),
        attachment="",
    )


def _open_source(source: _Source) -> tuple[int, os.stat_result]:
    """Open no-follow, non-blocking, regular-only; refuse a file over the cap."""
    try:
        fd = sync.open_for_read(source.path, source.scope, nonblock=True)
    except sync.SyncRefused as refusal:
        raise _refuse("unsafe_path", refusal.message) from None
    except OSError as error:
        raise _refuse(
            "missing", f"{source.name} is not readable on this device ({error})"
        ) from None
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):  # pragma: no cover - opened_regular refuses first
        os.close(fd)
        raise _refuse("unsafe_path", f"{source.name} is not a regular file")
    if info.st_size > FILE_MAX_BYTES:
        os.close(fd)
        raise _refuse(
            "too_large",
            f"{source.name} is {info.st_size} bytes; one pull carries at most "
            f"{FILE_MAX_BYTES} bytes",
        )
    return fd, info


def _read_span(fd: int, offset: int, limit: int) -> bytes:
    """``limit`` bytes at ``offset``. A function so a corruption control can wrap it."""
    chunks: list[bytes] = []
    remaining = limit
    position = offset
    while remaining > 0:
        block = os.pread(fd, remaining, position)
        if not block:
            break
        chunks.append(block)
        remaining -= len(block)
        position += len(block)
    return b"".join(chunks)


def _hash_fd(fd: int, length: int) -> str:
    """sha256 of the first ``length`` bytes of an open file, streamed in 1 MiB reads."""
    digest = hashlib.sha256()
    position = 0
    while position < length:
        block = os.pread(fd, min(1 << 20, length - position), position)
        if not block:
            break
        digest.update(block)
        position += len(block)
    return "sha256:" + digest.hexdigest()


def _stamp_matches(info: os.stat_result, frame: dict[str, Any]) -> bool:
    try:
        return int(frame["expect_bytes"]) == info.st_size and int(frame["expect_mtime_ns"]) == int(
            info.st_mtime_ns
        )
    except (KeyError, TypeError, ValueError):
        return False


def serve_file(root: Path, session_id: str, frame: dict[str, Any]) -> dict[str, Any]:
    """The owner's answer to one ``net_file_get`` phase. Raises :class:`MeshRefusal`."""
    phase = str(frame.get("phase") or "")
    source = _resolve_source(Path(root), session_id, frame)
    fd, info = _open_source(source)
    try:
        if phase == "stat":
            if info.st_size == 0:
                raise _refuse("empty_file", f"{source.name} is empty; there is nothing to pull")
            digest = _hash_fd(fd, info.st_size)
            if source.attachment and digest[len("sha256:") :][:32] != source.attachment:
                # The store's own name is a digest of its content; a file that no longer
                # hashes to it is damage, and serving it would launder the damage into a
                # verified copy on the requester.
                raise _refuse(
                    "corrupt_source",
                    f"attachment {source.attachment} on this device no longer matches its "
                    "digest, so it is not served",
                )
            return {
                "name": source.name,
                "bytes": info.st_size,
                "sha256": digest,
                "attachment": digest[len("sha256:") :][:32],
                "mime": source.mime,
                "mtime_ns": int(info.st_mtime_ns),
            }
        # EVERY LATER PHASE CARRIES THE STAT'S STAMP. Size + mtime is the session
        # copy's own cheap staleness check: a file rewritten mid-pull is refused
        # by name here, and a same-size same-mtime rewrite still fails the
        # requester's whole-file digest.
        if not _stamp_matches(info, frame):
            raise _refuse(
                "stale_file",
                f"{source.name} changed while it was being pulled; pulling again picks up "
                "the new version",
            )
        if phase == "fetch":
            try:
                offset = max(0, int(frame.get("offset") or 0))
                limit = min(FILE_CHUNK_BYTES, max(0, int(frame.get("limit") or FILE_CHUNK_BYTES)))
            except (TypeError, ValueError):
                raise _refuse("bad_request", "a fetch needs an integer offset and limit") from None
            span = _read_span(fd, offset, limit)
            return {
                "offset": offset,
                "bytes": len(span),
                "data": base64.b64encode(span).decode("ascii"),
                "eof": offset + len(span) >= info.st_size,
                "chunk_digest": sync.sha256_bytes(span),
                "file_bytes": info.st_size,
            }
        if phase == "verify":
            try:
                prefix = int(frame.get("prefix_bytes") or 0)
            except (TypeError, ValueError):
                raise _refuse("bad_request", "a verify needs an integer prefix_bytes") from None
            if prefix <= 0 or prefix > info.st_size:
                return {"match": False}
            return {"match": _hash_fd(fd, prefix) == str(frame.get("prefix_digest") or "")}
        raise _refuse("bad_request", f"unknown file phase {phase!r}")
    finally:
        os.close(fd)


def make_handler(server: "RelayServer") -> Callable[["PeerLink", dict[str, Any]], dict[str, Any]]:
    """The ``net_file_get`` handler. It issues no request over the link it answers."""

    def _handle(_link: "PeerLink", frame: dict[str, Any]) -> dict[str, Any]:
        session_id = str(frame.get("session_id") or "")
        # THE SESSION ID IS REQUIRED HERE, not only scoped by the chokepoint:
        # ``_session_scope`` checks ownership of a session the frame NAMES and
        # returns nothing for a frame that names none, so a frame without one
        # must not reach a path built from an empty id.
        if not session_id:
            raise _refuse("bad_request", "a file pull names the session it reads from")
        return serve_file(server.root, session_id, frame)

    return _handle


# ---------------------------------------------------------------------------
# The requester's half: pull, stage, verify, land
# ---------------------------------------------------------------------------


class _IntegrityError(Exception):
    """The bytes received do not match what the owner announced."""


def _part_lock(root: Path, sha: str) -> threading.Lock:
    key = (str(Path(root)), sha)
    with _PART_LOCKS_GUARD:
        lock = _PART_LOCKS.get(key)
        if lock is None:
            lock = _PART_LOCKS[key] = threading.Lock()
        return lock


def staging_root(root: Path) -> Path:
    return Path(root).joinpath(*STAGING_PARTS)


def _sweep_stale_parts(root: Path, scope: sync.WriteScope, *, now: float | None = None) -> None:
    """Delete staged parts nobody resumed within ``PART_MAX_AGE_S``. Best effort."""
    directory = staging_root(root)
    cutoff = (time.time() if now is None else now) - PART_MAX_AGE_S
    try:
        entries = list(os.scandir(directory))
    except OSError:
        return
    for entry in entries:
        try:
            if entry.name.endswith(".part") and entry.stat(follow_symlinks=False).st_mtime < cutoff:
                sync.unlink_at(directory, entry.name, scope)
        except (OSError, sync.SyncRefused):
            continue


def _open_part(root: Path, scope: sync.WriteScope, sha: str) -> tuple[int, Path]:
    """Open (creating) ``<staging>/<sha256>.part`` no-follow, under the relay's root."""
    directory = staging_root(root)
    name = f"{sha[len('sha256:') :]}.part"
    parent = sync.open_directory(scope, scope.relative_in(directory), name)
    try:
        try:
            fd = os.open(
                name,
                os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0),
                0o600,
                dir_fd=parent,
            )
        except OSError as error:
            raise _refuse(
                "staging_failed", f"the staging file could not be opened safely: {error}"
            ) from None
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise _refuse("staging_failed", "the staging file is not a regular file")
        return fd, directory / name
    finally:
        os.close(parent)


def _drop_part(root: Path, scope: sync.WriteScope, part: Path) -> None:
    try:
        sync.unlink_at(staging_root(root), part.name, scope)
    except (OSError, sync.SyncRefused):
        logger.debug("file pull: could not remove staged part %s", part, exc_info=True)


class _Asker:
    """``net_file_get`` requests over one link, refusals normalised to MeshRefusal."""

    def __init__(
        self,
        server: "RelayServer",
        link: "PeerLink",
        label: str,
        session_id: str,
        source: dict[str, str],
    ) -> None:
        self.server = server
        self.link = link
        self.label = label
        self.session_id = session_id
        self.source = source
        #: How many bytes are staged right now, so a lost link says what survives.
        self.staged = 0

    def ask(self, **fields: Any) -> dict[str, Any]:
        reply = self.link.request(
            {
                "op": "net_file_get",
                "req": self.server._next_relay_req(),  # noqa: SLF001 — the relay's req counter
                "locality": "remote",
                "session_id": self.session_id,
                **self.source,
                **fields,
            },
            timeout=self.server.slow_request_timeout("net_file_get"),
        )
        if reply is None:
            kept = (
                f"; {self.staged} bytes are staged and the next pull resumes from there"
                if self.staged
                else ""
            )
            raise _refuse(
                "peer_unreachable", f"{self.label} stopped answering during the pull{kept}"
            )
        if reply.get("op") == "error":
            raise _refuse(
                str(reply.get("code") or "peer_refused"),
                str(reply.get("message") or f"{self.label} refused the pull"),
            )
        detail = reply.get("detail")
        if reply.get("op") != "ack" or not isinstance(detail, dict):
            raise _refuse("peer_refused", f"{self.label} answered something this build cannot read")
        return detail


def _validated_stat(detail: dict[str, Any], label: str) -> dict[str, Any]:
    sha = str(detail.get("sha256") or "")
    try:
        size = int(detail.get("bytes") or -1)
        mtime_ns = int(detail.get("mtime_ns") or 0)
    except (TypeError, ValueError):
        size, mtime_ns = -1, 0
    if not _SHA256.fullmatch(sha) or size <= 0:
        raise _refuse("peer_refused", f"{label} described the file in a shape this build refuses")
    if size > FILE_MAX_BYTES:
        # The requester's own belt: the owner refuses this too, but a cap is only a
        # cap if the side that allocates the disk enforces it.
        raise _refuse(
            "too_large", f"the file is {size} bytes; one pull carries at most {FILE_MAX_BYTES}"
        )
    name = os.path.basename(str(detail.get("name") or "")) or "file"
    mime = str(detail.get("mime") or "") or "application/octet-stream"
    return {"sha256": sha, "bytes": size, "mtime_ns": mtime_ns, "name": name, "mime": mime}


def _resume_point(asker: _Asker, fd: int, meta: dict[str, Any]) -> int:
    """How many staged bytes are the owner's own prefix; 0 truncates and starts over."""
    have = os.fstat(fd).st_size
    if have <= 0:
        return 0
    if have <= meta["bytes"]:
        reply = asker.ask(
            phase="verify",
            prefix_bytes=have,
            prefix_digest=_hash_fd(fd, have),
            expect_bytes=meta["bytes"],
            expect_mtime_ns=meta["mtime_ns"],
        )
        if reply.get("match") is True:
            return have
    os.ftruncate(fd, 0)
    return 0


def _transfer(asker: _Asker, fd: int, meta: dict[str, Any], start: int) -> None:
    """Fetch from ``start`` to the end, checking every chunk, then the whole file."""
    total = meta["bytes"]
    offset = start
    asker.staged = start
    while offset < total:
        chunk = asker.ask(
            phase="fetch",
            offset=offset,
            limit=min(FILE_CHUNK_BYTES, total - offset),
            expect_bytes=total,
            expect_mtime_ns=meta["mtime_ns"],
        )
        try:
            data = base64.b64decode(str(chunk.get("data") or ""), validate=True)
        except ValueError:
            raise _IntegrityError("a chunk was not valid base64") from None
        if (
            int(chunk.get("offset", -1)) != offset
            or len(data) != int(chunk.get("bytes", -1))
            or not data
            or sync.sha256_bytes(data) != str(chunk.get("chunk_digest") or "")
            or offset + len(data) > total
        ):
            raise _IntegrityError(f"the chunk at offset {offset} failed its own digest")
        os.pwrite(fd, data, offset)
        offset += len(data)
        asker.staged = offset
    os.fsync(fd)
    if os.fstat(fd).st_size != total or _hash_fd(fd, total) != meta["sha256"]:
        raise _IntegrityError("the whole file did not hash to the announced digest")


def pull_file(
    server: "RelayServer",
    *,
    peer: str,
    session_id: str,
    path: str = "",
    attachment: str = "",
) -> dict[str, Any]:
    """Pull one file from ``peer``'s session into THIS device's attachment store.

    Returns the receipt ``{ok, peer, device_id, session_id, name, path, bytes,
    sha256, attachment, mime, verified, resumed_bytes}``; raises :class:`MeshRefusal`
    for every refusal, its code naming the rule. ``path`` in the receipt is the
    store file the CLI copies to ``--out``.
    """
    from local_operator.session.attachments import ATTACHMENTS_DIRNAME, AttachmentStore

    if not session_id:
        raise _refuse("bad_request", "a file pull names the session it reads from")
    if bool(path) == bool(attachment):
        raise _refuse(
            "bad_request",
            "a file pull names exactly one source: a scratchpad:// path or an attachment digest",
        )
    device_id = server._resolve_peer(peer)  # noqa: SLF001 — the family's one name resolver
    label = server._peer_label(peer)  # noqa: SLF001
    link = server._ensure_link(device_id)  # noqa: SLF001 — the one dial seam
    if link is None:
        raise _refuse(
            "peer_unreachable",
            f"{label} cannot be reached from this device right now, so no " "file was pulled",
        )
    if wire.MESH_FILE_V1 not in link.capabilities:
        # NEVER A FALLBACK: the route this replaces (base64 through the model) is the
        # one that corrupted bytes, so an old peer gets a named remedy, not a detour.
        raise _refuse(
            "predates_file_transfer",
            f"{label} does not support file transfer; update {label} (run `lop update` on "
            "it) to pull files from its sessions",
        )
    source = {"path": path} if path else {"attachment": attachment}
    asker = _Asker(server, link, label, session_id, source)
    meta = _validated_stat(asker.ask(phase="stat"), label)
    root = Path(server.root)
    scope = sync.write_scope(root, staging_root(root))
    _sweep_stale_parts(root, scope)
    with _part_lock(root, meta["sha256"]):
        resumed = 0
        # Named before the loop as well as by ``_open_part``: the same spelling, so the
        # landing below always refers to the part this pull wrote.
        part = staging_root(root) / f"{meta['sha256'][len('sha256:') :]}.part"
        for attempt in (1, 2):
            fd, part = _open_part(root, scope, meta["sha256"])
            try:
                start = _resume_point(asker, fd, meta)
                if attempt == 1:
                    resumed = start
                _transfer(asker, fd, meta, start)
            except _IntegrityError as error:
                os.close(fd)
                _drop_part(root, scope, part)
                if attempt == 1:
                    # ONE RETRY FROM ZERO: a single damaged chunk (or a damaged
                    # resumed prefix that still matched a forged verify) should not
                    # need a human; a second failure means the damage is not
                    # transient, and refusing is the only honest answer.
                    logger.info(
                        "file pull from %s failed verification (%s); retrying", label, error
                    )
                    resumed = 0
                    continue
                raise _refuse(
                    "integrity_failed",
                    f"the bytes from {label} did not match the digest it announced "
                    f"({meta['sha256']}); nothing was saved. Pull again; if it repeats, the "
                    f"file on {label} is changing or the link is damaging data",
                ) from None
            except MeshRefusal as refusal:
                os.close(fd)
                if refusal.code != "peer_unreachable":
                    # A refusal mid-pull (the file changed, the session stopped
                    # referencing it, the member lost view) makes the staged prefix
                    # worthless or no longer ours to hold: delete it. A LOST LINK is
                    # the one case where the part is kept, because resuming it is
                    # the whole point.
                    _drop_part(root, scope, part)
                raise
            except BaseException:
                os.close(fd)
                raise
            os.close(fd)
            break
        store = AttachmentStore(root / ATTACHMENTS_DIRNAME)
        # The store re-hashes while adopting: its own gate is what makes a file
        # visible under a content name, and it must not rest on a caller's check.
        ref = store.adopt_file(part, meta["mime"], expected_sha256=meta["sha256"])
        if ref is None:
            _drop_part(root, scope, part)
            raise _refuse(
                "landing_failed",
                "the pulled file verified but could not be filed in this device's attachment "
                "store; nothing was saved",
            )
    return {
        "ok": True,
        "peer": label,
        "device_id": device_id,
        "session_id": session_id,
        "name": meta["name"],
        "path": str(store.root / f"{ref.digest}.bin"),
        "bytes": meta["bytes"],
        "sha256": meta["sha256"],
        "attachment": ref.digest,
        "mime": meta["mime"],
        "verified": True,
        "resumed_bytes": resumed,
    }


def local_pull_handler(server: "RelayServer") -> Callable[[dict[str, Any]], dict[str, Any]]:
    """The ``session_file_pull`` local op: the CLI's frame carries names and paths only."""

    def _handle(frame: dict[str, Any]) -> dict[str, Any]:
        return pull_file(
            server,
            peer=str(frame.get("peer") or ""),
            session_id=str(frame.get("session_id") or ""),
            path=str(frame.get("path") or ""),
            attachment=str(frame.get("attachment") or ""),
        )

    return _handle


def install(server: "RelayServer") -> None:
    """Register ``net_file_get`` (slow: a stat hashes up to 256 MiB) and the local verb."""
    server.register_ops(
        {"net_file_get": make_handler(server)},
        local_handlers={"session_file_pull": local_pull_handler(server)},
        slow={"net_file_get": FILE_OP_DEADLINE_S},
    )
