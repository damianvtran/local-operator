"""``net_mentioned_file``: serve a transcript-named file, default OFF (P3 serving).

WHY THIS MODULE IS THIN. Every authority decision already lives elsewhere and
is re-made on each phase: P1's live account/device binding
(``account_binding.verify_live``), P2's exact transcript proof
(``file_provenance.lookup_proof``) and B1's audited seal/prepare
(``mentioned_file_read``). This module only (1) refuses a malformed frame before
any lookup, (2) derives the OWNER-side account inputs through
``file_device_approval._owner_serving_account`` — never from the approval slot,
see that function for the trap — and (3) bridges onto the relay's one page loop.

REACHABILITY. ``files.install`` imports this module only when the operator set
``mentioned_file_serving: true``; with the shipped default nothing here, and
nothing in ``mentioned_file_read``, is imported. The dormancy guard in
``tests/unit/network/test_mentioned_file_read.py`` pins exactly that edge.

THE FRAME CARRIES ROW IDENTITY ONLY. A peer names a session, a transcript row,
its provenance id and proof digest; it never names a path. Any path-like key is
refused, so there is no way to turn this op into "read the file I name".

Replies reuse ``net_file_get``'s shapes (stat/fetch/verify) so a puller needs no
new decoder; the stat additionally carries ``version``, the sealed commitment
every later phase must quote (the owner serves the sealed snapshot, not the
live file, and a stat/mtime stamp cannot name that).
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import json
import re
import secrets
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from local_operator.network import account_binding as B
from local_operator.network import file_device_approval as F
from local_operator.network import files
from local_operator.network import mentioned_file_read as M
from local_operator.network.types import MeshRefusal
from local_operator.session import file_provenance as P

if TYPE_CHECKING:
    from local_operator.network.relay import PeerLink, RelayServer

OP = "net_mentioned_file"

#: How long one request waits on the page loop. The relay's own owner-side
#: deadline for this op (``files.FILE_OP_DEADLINE_S``, registered as its slow
#: deadline) already bounds the handler; waiting LONGER here would only hold a
#: slow worker after the relay has answered the peer. Not a new measured cap:
#: the same bound ``net_file_get`` runs under, for the same work (one stat hash
#: of at most ``FILE_MAX_BYTES``, or one chunk).
MENTIONED_FILE_OP_TIMEOUT_S = files.FILE_OP_DEADLINE_S

_HEX32 = re.compile(r"[a-f0-9]{32}\Z")
_DIGEST = re.compile(r"sha256:[a-f0-9]{64}\Z")
_SEALED = re.compile(r"sealed:sha256:[a-f0-9]{64}\Z")
_SESSION = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_ROW = re.compile(r"[A-Za-z0-9_:.-]{1,256}\Z")
#: Keys that name a location. Refused by name (plus anything with ``/`` or
#: ``path`` in it) rather than ignored: a peer that sends one believes it can
#: choose what is read, and a silent ignore would hide that misuse.
_PATH_KEYS = frozenset({"path", "name", "file", "out", "filename", "dest", "land"})


def _bad() -> NoReturn:
    raise F._refuse("mentioned_file_bad_request")


def _unavailable() -> NoReturn:
    raise F._refuse("mentioned_file_unavailable")


def _int(frame: dict[str, Any], key: str, default: int | None) -> int | None:
    value = frame.get(key, default)
    if value is None:
        return None
    # bool is an int subclass; ``True`` is not an offset.
    if type(value) is not int or value < 0:
        _bad()
    return value


def _parse(frame: dict[str, Any]) -> dict[str, Any]:
    """Validate the whole frame BEFORE any proof, account or file lookup."""
    if not isinstance(frame, dict) or isinstance(frame.get("frame"), dict):
        _bad()
    for key in frame:
        if not isinstance(key, str) or key in _PATH_KEYS or "/" in key or "path" in key:
            _bad()
    text: dict[str, str] = {}
    for key in (
        "session_id",
        "provenance_id",
        "source_row_id",
        "proof_digest",
        "transfer_nonce",
        "phase",
    ):
        value = frame.get(key)
        if type(value) is not str:
            _bad()
        text[key] = value
    version = frame.get("file_version", "")
    prefix = frame.get("prefix_digest")
    if (
        not _SESSION.match(text["session_id"])
        or not _HEX32.match(text["provenance_id"])
        or not _ROW.match(text["source_row_id"])
        or not _DIGEST.match(text["proof_digest"])
        or not _HEX32.match(text["transfer_nonce"])
        or text["phase"] not in {"stat", "fetch", "verify"}
        or type(version) is not str
        or (version and not _SEALED.match(version))
        or (prefix is not None and (type(prefix) is not str or not _DIGEST.match(prefix)))
    ):
        _bad()
    offset = _int(frame, "offset", 0)
    length = _int(frame, "length", None)
    limit = _int(frame, "limit", None)
    phase = text["phase"]
    # Only a stat may establish; later phases quote the version it sealed.
    if phase != "stat" and not version:
        _bad()
    if phase == "stat" and (offset or length or limit):
        _bad()
    if phase == "verify" and (offset or prefix is None):
        _bad()
    return {
        **text,
        "file_version": version,
        "offset": offset,
        "length": length,
        "limit": limit,
        "prefix_digest": prefix,
    }


class MentionedFileServing:
    """Owner-side handler state: the sealed versions each live link established.

    A SealedVersion cannot leave the owner process, so the peer quotes its
    commitment and this map resolves it. Versions are scoped to the link that
    established them and dropped when it closes (``forget_link``): a reconnect
    re-establishes, which re-runs the audited establishment, rather than
    inheriting a version admitted under another link.
    """

    def __init__(self, server: "RelayServer") -> None:
        self.server = server
        self._lock = threading.Lock()
        self._versions: dict[str, tuple[Any, dict[str, tuple[Any, str, int]]]] = {}

    def forget_link(self, link: "PeerLink", _reason: str = "") -> None:
        with self._lock:
            held = self._versions.get(link.link_id)
            if held is not None and held[0] is link:
                del self._versions[link.link_id]

    def _known(self, link: "PeerLink", commitment: str) -> tuple[Any, str, int]:
        with self._lock:
            held = self._versions.get(link.link_id)
            if held is None or held[0] is not link or commitment not in held[1]:
                _unavailable()
            return held[1][commitment]

    def _remember(self, link: "PeerLink", version: Any, name: str, mtime_ns: int) -> None:
        with self._lock:
            held = self._versions.get(link.link_id)
            if held is None or held[0] is not link:
                held = (link, {})
                self._versions[link.link_id] = held
            held[1][version.commitment] = (version, name, mtime_ns)

    def _run(self, coroutine: Any) -> Any:
        """Bridge onto the relay's ONE page loop, never ``asyncio.run`` per call.

        Same discipline as ``RelayServer._stored_page_read``: choosing the loop
        and submitting to it is one critical section of ``_page_loop_lock``, so
        a concurrent ``stop`` cannot close the loop between the two.
        """
        server = self.server
        try:
            with server._page_loop_lock:  # noqa: SLF001 — the relay's own loop
                loop = server._page_loop_locked()  # noqa: SLF001
                future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        except BaseException:
            coroutine.close()
            raise
        try:
            return future.result(timeout=MENTIONED_FILE_OP_TIMEOUT_S)
        except concurrent.futures.TimeoutError:
            # Cancel before the caller closes the auth store the coroutine uses.
            future.cancel()
            raise F._refuse("mentioned_file_owner_timeout") from None

    def handle(self, link: "PeerLink", frame: dict[str, Any]) -> dict[str, Any]:
        request = _parse(frame)
        version: M.SealedVersion | None = None
        name, mtime_ns = "", 0
        if request["file_version"]:
            version, name, mtime_ns = self._known(link, request["file_version"])
        # _parse guarantees fetch/verify quote a version; restate it for the
        # size checks below rather than trusting that across two functions.
        if request["phase"] != "stat" and version is None:
            _bad()
        root = F._link_root(link)
        if Path(self.server.root).resolve() != root:
            raise F._refuse("file_approval_listener_required")
        mesh = F._mesh(link, root)
        offset = request["offset"]
        length = 0
        if request["phase"] == "fetch" and version is not None:
            wanted = request["limit"] if request["limit"] is not None else request["length"]
            wanted = files.FILE_CHUNK_BYTES if wanted is None else wanted
            if offset > version.size:
                _unavailable()
            length = min(wanted, files.FILE_CHUNK_BYTES, version.size - offset)
        elif request["phase"] == "verify" and version is not None:
            length = request["length"] or 0
            if length <= 0 or length > version.size:
                # net_file_get's own answer for an impossible prefix: no read.
                return {"match": False}
        read = B.ReadBinding(
            owner_key=mesh["owner"],
            network_id=mesh["network_id"],
            network_epoch=mesh["network_epoch"],
            link_id=link.link_id,
            transcript_hash=link.result.transcript_hash,
            request_id="req_" + secrets.token_hex(16),
            session_id=request["session_id"],
            source_row_id=request["source_row_id"],
            provenance_id=request["provenance_id"],
            proof_digest=request["proof_digest"],
            file_version=request["file_version"] or "establish",
            phase=request["phase"],
            offset=offset,
            length=length,
            transfer_nonce=request["transfer_nonce"],
        )
        # The auth store stays open for the WHOLE request: verify_live reads
        # the native row through it on every await inside establish/prepare.
        with F._own_auth(root) as auth:
            account = F._owner_serving_account(link, auth, root)
            reader = M.OwnerMentionedFileReader(
                incoming_link=link,
                auth=auth,
                owner_account=B.ExactNativeAccountSelection(credential_id=account.credential_id),
                evidence=B.OwnerFileEvidenceReference(association=account.association),
                required_subject=B.CanonicalSubject(
                    issuer=account.issuer, opaque_id=account.opaque_id
                ),
            )
            try:
                if version is None:
                    version, prepared = self._run(reader.establish(M.begin_establishment(read)))
                    name, mtime_ns = self._describe(root, read, version)
                else:
                    prepared = self._run(reader.prepare(version, read))
            except MeshRefusal:
                # A relogin mid-request surfaces inside verify_live as a
                # generic no-active-slot refusal; name it for what it is.
                account.require_unchanged(auth, root)
                raise
            account.require_unchanged(auth, root)
        if not request["file_version"]:
            self._remember(link, version, name, mtime_ns)
        return self._reply(request, prepared, name, mtime_ns)

    @staticmethod
    def _describe(root: Path, read: B.ReadBinding, version: Any) -> tuple[str, int]:
        """Display name and the SEALED source mtime, committed at establishment."""
        proof = P.lookup_proof(root, read.session_id, read.provenance_id)
        if proof is None:
            _unavailable()
        try:
            # The seal's source_stamp is file_provenance._identity + st_mode:
            # (dev, ino, ctime_ns, mtime_ns, size, nlink, mode).
            mtime_ns = int(json.loads(version._seal_bytes)["source_stamp"][3])
        except (ValueError, KeyError, IndexError, TypeError):
            _unavailable()
        return Path(proof["file"]["path"]).name, mtime_ns

    @staticmethod
    def _reply(request: dict[str, Any], prepared: Any, name: str, mtime_ns: int) -> dict[str, Any]:
        if request["phase"] == "stat":
            return {
                "name": name,
                "bytes": prepared.size,
                "sha256": prepared.whole_digest,
                "attachment": prepared.whole_digest[len("sha256:") :][:32],
                "mime": "application/octet-stream",
                "mtime_ns": mtime_ns,
                "version": prepared.version,
            }
        if request["phase"] == "fetch":
            offset = prepared.read_binding.offset
            return {
                "offset": offset,
                "bytes": len(prepared.payload),
                "data": base64.b64encode(prepared.payload).decode("ascii"),
                "eof": offset + len(prepared.payload) >= prepared.size,
                "chunk_digest": prepared.span_digest,
                "file_bytes": prepared.size,
            }
        return {"match": prepared.span_digest == request["prefix_digest"]}


def install(server: "RelayServer") -> None:
    """Register the op. Called ONLY from ``files.install``'s enabled branch."""
    serving = MentionedFileServing(server)
    server.register_ops(
        {OP: serving.handle},
        slow={OP: files.FILE_OP_DEADLINE_S},
        on_link_closed={OP: serving.forget_link},
    )
