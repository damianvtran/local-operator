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


#: The credential-vocabulary rule ids (``supplements/denylist.py``) as sentences.
#: The id itself is the denylist's internal name ("name", "token-or-secret-name"),
#: which a person cannot read without the source (#3794 UX U3). The code stays
#: ``sensitive``; the rule id does not cross the link (a refusal frame carries a
#: code and a sentence only), so the sentence is what explains it.
_SENSITIVE_WHY: dict[str, str] = {
    "name": "it is named like a private key or credential file",
    "name-prefix": "it is named like a private key or credential file",
    "suffix": "its extension is one used for keys and certificates",
    "component": "it sits in a folder used for credentials",
    "credential-pattern": "its name looks like a credential file",
    "database": "it looks like a local database, which can hold saved credentials",
    "token-or-secret-name": "its name looks like a credential (token, secret, key)",
    "compose-file": "it is a container compose file, which often carries secrets",
    "gh-hosts": "it is a GitHub CLI login file",
    "docker-config": "it is a Docker login file",
    "gcloud": "it is a Google Cloud login file",
    "terraform": "it is a Terraform login file",
    "tool-credential-config": "it is a tool's credential file",
    "browser-store": "it is a browser's cookie or password store",
    "config-dir": "it is inside that device's own configuration",
    "scratchpad": "it is in another session's scratchpad",
}


def _sensitive_sentence(path_ref: str, rule: str) -> str:
    why = _SENSITIVE_WHY.get(rule, "its name looks like a credential")
    return (
        f"{path_ref} is not served: {why}, and credential-looking files never leave a "
        "device over the mesh. If it is not a secret, have the session save it under "
        "another name and pull that"
    )


def human_bytes(count: int) -> str:
    """``3.0 MiB``: the text surfaces' size; exact integers stay in ``--json``."""
    value = float(count)
    for unit in ("bytes", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{int(value)} bytes" if unit == "bytes" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{count} bytes"  # pragma: no cover - the loop always returns


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
    #: What the requester asked for, as they wrote it: every refusal names THIS, never
    #: the owner's absolute path.
    label: str = ""


def _resolve_source(
    root: Path, session_id: str, frame: dict[str, Any], owner: str = "this device"
) -> _Source:
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
                f"attachment:{attachment} is not part of session {session_id} on {owner}, so "
                "it is not served; only files that conversation references can be pulled",
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
            scope=_read_scope(Path(root), store_root, f"attachment:{attachment}"),
            name=f"{attachment}{extension}",
            mime=mime,
            attachment=attachment,
            label=f"attachment:{attachment}",
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
            raise _refuse("sensitive", _sensitive_sentence(path_ref, reason))
    return _Source(
        path=target.path,
        scope=_read_scope(Path(root), pad, path_ref),
        name=target.path.name,
        mime=_mime_for(target.path.name),
        attachment="",
        label=path_ref,
    )


def _read_scope(root: Path, allowed: Path, label: str) -> sync.WriteScope:
    """The containment scope for a READ, its refusal in read words with no owner path.

    ``sync.write_scope`` raises ``SyncRefused`` with the write side's sentence and the
    absolute path ("Sync member '/…/scratchpad' was not written: …"). Escaping as-is it
    reached the requester through the relay's generic error path, naming this device's
    filesystem layout to a read member (#3794 QA Q3, review R1-2).
    """
    try:
        return sync.write_scope(root, allowed)
    except sync.SyncRefused:
        raise _refuse(
            "unsafe_path",
            f"{label} is not served: the folder it is in is a link on that device, not a "
            "plain folder",
        ) from None


def _not_found(label: str, session_id: str, owner: str) -> MeshRefusal:
    return _refuse(
        "not_found",
        f"{label} does not exist in session {session_id} on {owner}; check the path the "
        "session reported (paths are case-sensitive)",
    )


def _open_source(source: _Source, session_id: str, owner: str) -> tuple[int, os.stat_result]:
    """Open no-follow, non-blocking, regular-only, single-link; refuse over the cap.

    NOT FOUND IS ITS OWN ANSWER (#3794 UX U2): a typo or a file the agent has not
    written yet is the refusal people meet most, and calling it ``unsafe_path`` sent
    an agent branching on the code down the confinement-violation road. The ``lstat``
    only CLASSIFIES; the open below is what decides, and it never creates a parent.
    """
    label = source.label
    try:
        os.lstat(source.path)
    except (FileNotFoundError, NotADirectoryError):
        raise _not_found(label, session_id, owner) from None
    except OSError:
        pass  # the open below gives the real answer
    try:
        fd = sync.open_for_read(source.path, source.scope, nonblock=True)
    except sync.SyncRefused as refusal:
        if refusal.code == "missing":
            raise _not_found(label, session_id, owner) from None
        raise _refuse(
            "unsafe_path",
            f"{label} is not a regular file (a link, pipe or device), so it is not served",
        ) from None
    except OSError:
        raise _not_found(label, session_id, owner) from None
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):  # pragma: no cover - opened_regular refuses first
        os.close(fd)
        raise _refuse(
            "unsafe_path",
            f"{label} is not a regular file (a link, pipe or device), so it is not served",
        )
    if not source.attachment and info.st_nlink > 1:
        # A HARD LINK IS A SECOND NAME FOR SOMEONE ELSE'S BYTES (#3794 QA Q1). Every
        # check above judges the NAME inside the pad; a hard link to the network's
        # secrets file passes all of them and is a regular, no-follow file. Judged on
        # the OPENED fd, so a link made between a stat and the open cannot slip past.
        # Store blobs are exempt: the store writes them by rename, single-link, and a
        # digest names their content rather than a place.
        os.close(fd)
        raise _refuse(
            "hard_linked",
            f"{label} is not served: it is a hard link, so its bytes may belong to a "
            "file outside the session's scratchpad. Have the session save a copy rather "
            "than a link, and pull that",
        )
    if info.st_size > FILE_MAX_BYTES:
        os.close(fd)
        raise _refuse(
            "too_large",
            f"{label} is {human_bytes(info.st_size)}; one pull carries at most "
            f"{human_bytes(FILE_MAX_BYTES)}. Split or compress it in that session, then "
            "pull the parts",
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


def serve_file(
    root: Path, session_id: str, frame: dict[str, Any], owner: str = "this device"
) -> dict[str, Any]:
    """The owner's answer to one ``net_file_get`` phase. Raises :class:`MeshRefusal`.

    ``owner`` is this device's own NAME: these sentences are read on the requester,
    where "this device" means the one the person is typing on (#3794 UX U6).
    """
    phase = str(frame.get("phase") or "")
    source = _resolve_source(Path(root), session_id, frame, owner)
    fd, info = _open_source(source, session_id, owner)
    try:
        if phase == "stat":
            if info.st_size == 0:
                raise _refuse("empty_file", f"{source.label} is empty; there is nothing to pull")
            digest = _hash_fd(fd, info.st_size)
            if source.attachment and digest[len("sha256:") :][:32] != source.attachment:
                # The store's own name is a digest of its content; a file that no longer
                # hashes to it is damage, and serving it would launder the damage into a
                # verified copy on the requester.
                raise _refuse(
                    "corrupt_source",
                    f"attachment:{source.attachment} on {owner} no longer matches its "
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
                f"{source.label} changed while it was being pulled; pulling again picks up "
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
        if isinstance(frame.get("frame"), dict):
            # ONE SESSION ID, THE ONE THE CHOKEPOINT CHECKED (#3794 review R1-3).
            # ``_session_scope`` scopes a NESTED ``frame``'s session id when one is
            # present, and this handler serves the OUTER one; a frame carrying both
            # could name an owned session to the check and another directory here.
            # This op never nests, so a nested frame is malformed, not a variant.
            raise _refuse("bad_request", "a file pull carries no nested frame")
        session_id = str(frame.get("session_id") or "")
        # THE SESSION ID IS REQUIRED HERE, not only scoped by the chokepoint:
        # ``_session_scope`` checks ownership of a session the frame NAMES and
        # returns nothing for a frame that names none, so a frame without one
        # must not reach a path built from an empty id.
        if not session_id:
            raise _refuse("bad_request", "a file pull names the session it reads from")
        try:
            return serve_file(server.root, session_id, frame, server._own_label())  # noqa: SLF001
        except sync.SyncRefused:
            # Belt: a containment refusal this module did not translate must still
            # leave as a named read refusal, never the relay's generic error with
            # this device's absolute paths in it (#3794 QA Q3).
            raise _refuse("unsafe_path", "that file is not served from this session") from None

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
    try:
        parent = sync.open_directory(scope, scope.relative_in(directory), name)
    except sync.SyncRefused:
        raise _refuse(
            "staging_failed",
            "this device's staging folder for pulls is a link or not a folder, so nothing "
            "was pulled",
        ) from None
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
                f"; {human_bytes(self.staged)} arrived intact and is kept — run the same "
                "command again to resume"
                if self.staged
                else "; check it with `lop network peers` or `lop network doctor`"
            )
            raise _refuse(
                "peer_unreachable", f"{self.label} stopped answering during the pull{kept}"
            )
        if reply.get("op") == "error":
            raise self._owner_refusal(reply)
        detail = reply.get("detail")
        if reply.get("op") != "ack" or not isinstance(detail, dict):
            raise _refuse("peer_refused", f"{self.label} answered something this build cannot read")
        return detail

    def _owner_refusal(self, reply: dict[str, Any]) -> MeshRefusal:
        """The owner's refusal, re-voiced for the device the person is typing on.

        The owner's sentences say "this device" about ITSELF, which read on the
        requester as the device in front of the person (#3794 UX U6, QA Q5). The
        chokepoint's two refusals also cross WITHOUT a code — deliberately
        (``wire.refusal_frame``: the peer is not told which guard fired) — so their
        sentence is the only evidence, and it is matched here to give an agent the
        code the owner's audit log already recorded and a person the next step.
        """
        message = _printable(str(reply.get("message") or "it refused the pull"))
        code = str(reply.get("code") or "")
        own = _printable(self.server._own_label())  # noqa: SLF001 — this device's name
        if not code and "does not hold the 'view' capability" in message:
            record = self.server.store_view.network(self.link.network_id)
            network = _printable(str(getattr(record, "name", "") or "<network>"))
            return _refuse(
                "not_authorised",
                f"{self.label} refused: {own} lacks the 'view' capability there. An admin "
                f"on {self.label} can grant it: lop network member grant {network} {own} "
                "view",
            )
        if not code and "does not live on this device" in message:
            return _refuse(
                "session_not_found",
                f"{self.label} has no session {self.session_id}; list its sessions with "
                f"lop network sessions --peer {self.label}",
            )
        refusal = _refuse(code or "peer_refused", f"{self.label} refused: {message}")
        return refusal


def _printable(text: str) -> str:
    """``text`` with control characters escaped: a peer chose it, a terminal shows it."""
    return "".join(c if c.isprintable() else repr(c)[1:-1] for c in str(text))


#: Mime types this device files as the peer declared them (#3794 review R1-4): the
#: media the attachment routes already serve, plus passive text formats. Anything
#: else is filed as ``application/octet-stream``: the peer's word is a label, and a
#: later reader may serve the sidecar as ``Content-Type`` (``text/html`` for
#: arbitrary bytes is the case this closes).
_FILED_MIME_TYPES: frozenset[str] = frozenset(
    {
        "text/plain",
        "text/markdown",
        "text/csv",
        "application/json",
        "application/x-ndjson",
        "application/pdf",
    }
)


def _filed_mime(declared: str) -> str:
    from local_operator.media import (
        SUPPORTED_AUDIO_MIME_TYPES,
        SUPPORTED_IMAGE_MIME_TYPES,
        SUPPORTED_VIDEO_MIME_TYPES,
    )

    allowed = (
        _FILED_MIME_TYPES
        | SUPPORTED_IMAGE_MIME_TYPES
        | SUPPORTED_AUDIO_MIME_TYPES
        | SUPPORTED_VIDEO_MIME_TYPES
    )
    return declared if declared in allowed else "application/octet-stream"


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
            "too_large",
            f"the file is {human_bytes(size)}; one pull carries at most "
            f"{human_bytes(FILE_MAX_BYTES)}. Split or compress it in that session, then pull "
            "the parts",
        )
    name = _printable(os.path.basename(str(detail.get("name") or ""))) or "file"
    mime = _filed_mime(str(detail.get("mime") or ""))
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
    land: str = "store",
) -> dict[str, Any]:
    """Pull one file from ``peer``'s session into THIS device's attachment store.

    ``land="cache"`` files it in the viewer's bounded remote-files cache instead
    (``remote_files.cache``, docs/design/remote-files-parity.md §4): the desktop's
    fetch-on-miss lands there so eviction can never delete a local session's only
    copy from the authoritative store. ``"store"`` keeps S1's CLI receipt.

    Returns the receipt ``{ok, peer, device_id, session_id, name, path, bytes,
    sha256, attachment, mime, verified, resumed_bytes}``; raises :class:`MeshRefusal`
    for every refusal, its code naming the rule. ``path`` in the receipt is the
    store file the CLI copies to ``--out``.
    """
    from local_operator.session.attachments import ATTACHMENTS_DIRNAME, AttachmentStore

    if not session_id:
        raise _refuse("bad_request", "a file pull names the session it reads from")
    if land not in ("store", "cache"):
        raise _refuse("bad_request", f"a file pull lands in 'store' or 'cache', not {land!r}")
    if bool(path) == bool(attachment):
        raise _refuse(
            "bad_request",
            "a file pull names exactly one source: a scratchpad:// path or an attachment digest",
        )
    device_id = server._resolve_peer(peer)  # noqa: SLF001 — the family's one name resolver
    label = _printable(server._peer_label(peer))  # noqa: SLF001
    link = server._ensure_link(device_id)  # noqa: SLF001 — the one dial seam
    if link is None:
        raise _refuse(
            "peer_unreachable",
            f"{label} cannot be reached from this device right now, so no file was pulled; "
            "check it with `lop network peers` or `lop network doctor`",
        )
    if wire.MESH_FILE_V1 not in link.capabilities:
        # NEVER A FALLBACK: the route this replaces (base64 through the model) is the
        # one that corrupted bytes, so an old peer gets a named remedy, not a detour.
        raise _refuse(
            "predates_file_transfer",
            f"{label} does not support file transfer yet; update it from here with "
            f"`lop network update {label}` (or run `lop update` on it), then pull again",
        )
    source = {"path": path} if path else {"attachment": attachment}
    asker = _Asker(server, link, label, session_id, source)
    meta = _validated_stat(asker.ask(phase="stat"), label)
    root = Path(server.root)
    if land == "cache":
        from local_operator.remote_files.cache import cache_root, touch

        # A PATH source is keyed by the owner's stat sha256 only after this phase, so
        # the viewer cannot look the cache up first; the bytes may already be here
        # (content-addressed, verified when they landed). Skip the transfer then.
        digest = meta["sha256"][len("sha256:") :][:32]
        cached = cache_root(root) / f"{digest}.bin"
        if cached.is_file():
            touch(cached)
            return {
                "ok": True,
                "peer": label,
                "device_id": device_id,
                "session_id": session_id,
                "name": meta["name"],
                "path": str(cached),
                "bytes": meta["bytes"],
                "sha256": meta["sha256"],
                "attachment": digest,
                "mime": meta["mime"],
                "verified": True,
                "resumed_bytes": 0,
            }
    try:
        scope = sync.write_scope(root, staging_root(root))
    except sync.SyncRefused:
        raise _refuse(
            "staging_failed",
            "this device's staging folder for pulls is a link or not a folder, so nothing "
            "was pulled; remove network/staging/files under this device's config folder",
        ) from None
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
        if land == "cache":
            from local_operator.remote_files.cache import cache_root

            store = AttachmentStore(cache_root(root))
        else:
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
        if land == "cache":
            # An existing blob is not re-written by ``adopt_file``, so its mtime is
            # whenever it first landed: bump it, or the eviction pass that follows
            # this landing would treat the file just asked for as the oldest.
            from local_operator.remote_files.cache import touch

            touch(store.root / f"{ref.digest}.bin")
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
            land=str(frame.get("land") or "store"),
        )

    return _handle


def install(server: "RelayServer") -> None:
    """Register ``net_file_get`` (slow: a stat hashes up to 256 MiB) and the local verb."""
    from local_operator.network import file_index

    server.register_ops(
        {
            "net_file_get": make_handler(server),
            # The Files index rides this module's install so SLICE_MODULES is untouched
            # (remote-files-parity.md §2): same family, same view capability.
            "net_session_files": file_index.make_handler(server),
        },
        local_handlers={
            "session_file_pull": local_pull_handler(server),
            "peer_session_files": file_index.local_handler(server),
        },
        slow={
            "net_file_get": FILE_OP_DEADLINE_S,
            "net_session_files": file_index.INDEX_OP_DEADLINE_S,
        },
    )
