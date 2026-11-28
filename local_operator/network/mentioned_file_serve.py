"""``net_mentioned_file``: serve a transcript-named file, default OFF (P3 serving).

WHY THIS MODULE IS THIN. Every authority decision already lives elsewhere and
is re-made on each phase: P1's live account/device binding
(``account_binding.verify_live``), P2's exact transcript proof
(``file_provenance.lookup_proof``) and B1's audited seal/prepare
(``mentioned_file_read``). This module only (1) refuses a malformed frame before
any lookup, (2) derives the OWNER-side account inputs through
``file_device_approval._owner_serving_account`` — never from the approval slot,
see that function for the trap — (3) runs the reader on its own event loop, off
the relay's shared page loop, (4) keeps at most one sealed version per
transcript row per link, removing the ones it supersedes or that stop
preparing, with one establishment in flight per row per link, and (5) caps the
reader threads alive in this process at the relay's own slow-op bound, counted
until each thread's work truly ends (a timed-out reader keeps its slot).

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
live file, and a stat/mtime stamp cannot name that). Requests follow
``net_file_get`` too, with one addition: a verify names its prefix length as
``prefix_bytes`` (``net_file_get``'s spelling) or ``length`` (this op's own
fetch spelling); both mean the same, and naming two different values is refused.

WHAT A PEER LEARNS BEFORE P1. Until this link has a sealed version, the
requester has not been verified by P1, so EVERY refusal from the account
accessor, the reader or P1 itself (no native login, several, unknown inventory,
a relogin, no approval slot, revoked, ``/me`` failing) is answered with the same
generic ``mentioned_file_unavailable``; the typed code goes to the owner's log
(PR #4563 S1-2, S2-2). No proof lookup runs before P1 either, so whether the
named file changed since it was mentioned is not observable (S2-1). Once the
link holds a version (P1 verified it), refusals are answered typed. The only
typed answers before P1 are ``mentioned_file_owner_timeout`` and
``mentioned_file_owner_busy``: they describe the owner's load, never its
accounts or files.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import contextlib
import json
import logging
import os
import re
import secrets
import stat
import threading
from collections.abc import Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from local_operator.network import account_binding as B
from local_operator.network import file_device_approval as F
from local_operator.network import files
from local_operator.network import mentioned_file_read as M
from local_operator.network import sync
from local_operator.network.relay import (
    SLOW_OP_MAX_PENDING,
    SLOW_REPLY_MARGIN_S,
    slow_op_remaining_s,
)
from local_operator.network.types import MeshRefusal
from local_operator.session import file_provenance as P

if TYPE_CHECKING:
    from local_operator.network.relay import PeerLink, RelayServer

logger = logging.getLogger(__name__)

OP = "net_mentioned_file"

#: How long one request waits for its reader. STRICTLY BELOW the relay's own
#: deadline for this op (``files.FILE_OP_DEADLINE_S``, registered as its slow
#: deadline), by the relay's reply margin: the relay's overdue timer starts at
#: dispatch and answers the peer with a generic "did not finish" at exactly that
#: deadline, so a bridge waiting the same time could never deliver the typed
#: ``mentioned_file_owner_timeout`` (PR #4563 R1-3). Not a new measured cap:
#: derived from the bound ``net_file_get`` runs under for the same work. The
#: remaining slow-op budget lowers it further when the request queued.
MENTIONED_FILE_OP_TIMEOUT_S = files.FILE_OP_DEADLINE_S - SLOW_REPLY_MARGIN_S

#: Reader threads alive in this process (PR #4563 R2-1). The relay's slow-op
#: slots bound HANDLERS, and a handler returns at its wait budget while a reader
#: stalled in file IO (a hung network mount, a failing disk) keeps running with
#: an open source fd, an AuthStore connection and a half-written snapshot. So a
#: retrying peer could grow those without bound. A slot is taken BEFORE the
#: thread starts and given back only when its coroutine has really ended, so
#: abandoned readers count until they are done. Not a new measured cap: it is
#: the relay's own ``SLOW_OP_MAX_PENDING``, the bound ``net_file_get`` already
#: runs under for the same work. Process-wide on purpose: threads, fds and disk
#: are process resources, whichever relay instance started them.
_READERS = threading.BoundedSemaphore(SLOW_OP_MAX_PENDING)

_HEX32 = re.compile(r"[a-f0-9]{32}\Z")
_DIGEST = re.compile(r"sha256:[a-f0-9]{64}\Z")
_SEALED = re.compile(r"sealed:sha256:[a-f0-9]{64}\Z")
_SESSION = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_ROW = re.compile(r"[A-Za-z0-9_:.-]{1,256}\Z")
#: Keys that name a location. Refused by name (plus anything with ``/`` or
#: ``path`` in it) rather than ignored: a peer that sends one believes it can
#: choose what is read, and a silent ignore would hide that misuse.
_PATH_KEYS = frozenset({"path", "name", "file", "out", "filename", "dest", "land"})

#: (session_id, provenance_id, source_row_id, proof_digest): one transcript row's
#: named file. A link holds at most ONE sealed version per key.
_Key = tuple[str, str, str, str]


def _bad() -> NoReturn:
    raise F._refuse("mentioned_file_bad_request")


def _unavailable() -> NoReturn:
    raise F._refuse("mentioned_file_unavailable")


def _generic_refusal() -> MeshRefusal:
    """The reader's own refusal, byte for byte, so a peer cannot tell them apart."""
    try:
        M._refuse()
    except MeshRefusal as refusal:
        return refusal


def _owner_refusal(refusal: MeshRefusal, *, typed: bool) -> MeshRefusal:
    """A refusal as the PEER may see it (S1-2, S2-2).

    Before P1 has verified this requester, a typed code would tell any view
    member how many native logins the owner has, whether a relogin is under
    way, whether any approval slot exists, or whether the owner's ``/me`` check
    is failing: exactly one usable login is the only state that reaches P1.
    It is kept in the owner's log, and the peer gets the generic refusal.
    """
    if typed:
        return refusal
    logger.warning(
        "%s: refusal %s before P1 answered to the peer as mentioned_file_unavailable",
        OP,
        refusal.code,
    )
    return _generic_refusal()


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
    # ``net_file_get`` spells a verify's prefix length ``prefix_bytes`` (R1-6).
    prefix_bytes = _int(frame, "prefix_bytes", None)
    if prefix_bytes is not None:
        if phase != "verify" or (length is not None and length != prefix_bytes):
            _bad()
        length = prefix_bytes
    return {
        **text,
        "file_version": version,
        "offset": offset,
        "length": length,
        "limit": limit,
        "prefix_digest": prefix,
    }


@dataclass(frozen=True)
class _Held:
    """One sealed version a link may quote, plus what its stat reply commits to.

    ``incarnation`` is the owner's native login incarnation the version was
    established under (Q1-2): a later phase under another incarnation is a
    relogin, and is named so rather than surfacing as "no active approval".
    """

    version: Any  # M.SealedVersion; Any so this module never constructs one
    name: str
    mtime_ns: int
    incarnation: str


def _remove_version_directory(
    directory: Path, nonce: str, identity: tuple[int, ...] | list[int] | None
) -> bool:
    """Remove ONE ``mentioned-file-versions/<nonce>`` directory this module created.

    WHY: each establishment writes a full copy of the named file (up to
    ``FILE_MAX_BYTES``) into the session, and the reader keeps no sweeper
    (PR #4563 R1-2/S1-3). Only the exact nonce directory is touched: its parent is
    opened no-follow from the filesystem anchor, the entry must be a real
    directory (never a link) and, when known, have the identity the seal
    recorded; only its plain entries are unlinked, through descriptors, and a
    subdirectory makes it refuse. Any doubt leaves the directory in place, which
    is the pre-existing state, never a wider delete. Best effort: returns whether
    it was removed.
    """
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    if (
        not _HEX32.match(nonce)
        or directory.name != nonce
        or directory.parent.name != M._DIRECTORY
        or not directory_flag
        or not {os.open, os.stat, os.unlink, os.rmdir} <= os.supports_dir_fd
    ):
        return False
    parent = directory.parent
    try:
        scope = files._read_scope(Path(parent.anchor), parent, "mentioned file versions")
        parent_fd = sync.open_directory(
            scope, scope.relative_in(parent), "mentioned file versions", create=False
        )
    except (OSError, MeshRefusal, sync.SyncRefused):
        return False
    try:
        info = os.stat(nonce, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode) or (
            identity is not None and (info.st_dev, info.st_ino) != tuple(identity)
        ):
            return False
        fd = os.open(nonce, os.O_RDONLY | directory_flag | os.O_NOFOLLOW, dir_fd=parent_fd)
        try:
            opened = os.fstat(fd)
            if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                return False
            names = os.listdir(fd)
            for name in names:
                if stat.S_ISDIR(os.stat(name, dir_fd=fd, follow_symlinks=False).st_mode):
                    return False
            for name in names:
                os.unlink(name, dir_fd=fd)
        finally:
            os.close(fd)
        os.rmdir(nonce, dir_fd=parent_fd)
    except OSError:
        logger.debug("%s: could not remove snapshot %s", OP, nonce, exc_info=True)
        return False
    finally:
        os.close(parent_fd)
    return True


def _remove_version(version: Any) -> bool:
    return _remove_version_directory(version._directory, version.nonce, version._directory_identity)


def _wait_budget() -> float:
    remaining = slow_op_remaining_s()
    if remaining is None:
        return MENTIONED_FILE_OP_TIMEOUT_S
    return max(0.0, min(MENTIONED_FILE_OP_TIMEOUT_S, remaining - SLOW_REPLY_MARGIN_S))


class MentionedFileServing:
    """Owner-side handler state: the sealed versions each live link established.

    A SealedVersion cannot leave the owner process, so the peer quotes its
    commitment and this map resolves it. Versions are scoped to the link that
    established them and dropped, with their snapshot directories, when it
    closes (``forget_link``): a reconnect re-establishes, which re-runs the
    audited establishment, rather than inheriting a version admitted under
    another link. Within one link a re-stat of the same row REUSES the held
    version while it still prepares, so repeated stats cost one snapshot.
    """

    def __init__(self, server: "RelayServer") -> None:
        self.server = server
        self._lock = threading.Lock()
        self._versions: dict[str, tuple[Any, dict[_Key, _Held]]] = {}
        # One stat per (link, row) at a time (R2-2). Without it two concurrent
        # stats both establish and the second supersedes, and so removes, the
        # version the first had just handed its peer. A waiting stat then finds
        # the first one's version held and reuses it.
        self._flights: dict[tuple[str, _Key], concurrent.futures.Future[None]] = {}

    def forget_link(self, link: "PeerLink", _reason: str = "") -> None:
        with self._lock:
            held = self._versions.get(link.link_id)
            if held is None or held[0] is not link:
                return
            del self._versions[link.link_id]
        for entry in held[1].values():
            _remove_version(entry.version)

    def _known(self, link: "PeerLink", commitment: str) -> _Held:
        with self._lock:
            held = self._versions.get(link.link_id)
            if held is not None and held[0] is link:
                for entry in held[1].values():
                    if entry.version.commitment == commitment:
                        return entry
        _unavailable()

    def _held(self, link: "PeerLink", key: _Key) -> _Held | None:
        with self._lock:
            held = self._versions.get(link.link_id)
            if held is None or held[0] is not link:
                return None
            return held[1].get(key)

    async def _take_flight(
        self, link: "PeerLink", key: _Key, abandoned: threading.Event
    ) -> concurrent.futures.Future[None]:
        """Become the one stat of ``key`` on ``link``, waiting out any other first."""
        while True:
            with self._lock:
                flight = self._flights.get((link.link_id, key))
                if flight is None:
                    flight = concurrent.futures.Future()
                    self._flights[(link.link_id, key)] = flight
                    return flight
            await asyncio.wrap_future(flight)
            if abandoned.is_set():
                # Its peer was already told it timed out: do no work for nobody.
                raise _generic_refusal()

    def _land(self, link: "PeerLink", key: _Key, flight: concurrent.futures.Future[None]) -> None:
        with self._lock:
            if self._flights.get((link.link_id, key)) is flight:
                del self._flights[(link.link_id, key)]
        flight.set_result(None)

    def _drop(self, link: "PeerLink", key: _Key, entry: _Held) -> None:
        """Stop holding ``entry`` (it no longer prepares) and remove its directory.

        Q2-1: otherwise a re-stat whose re-establishment is refused (the file
        changed, so the proof no longer matches) left the dead copy held and on
        disk until the link closed. Only the exact entry still held is dropped,
        through the same identity-checked removal every other path uses.
        """
        with self._lock:
            held = self._versions.get(link.link_id)
            if held is None or held[0] is not link or held[1].get(key) is not entry:
                return
            del held[1][key]
        _remove_version(entry.version)

    def _install(self, link: "PeerLink", key: _Key, entry: _Held) -> None:
        """Hold ``entry`` as the link's ONE version for ``key``; remove what it replaces."""
        superseded: _Held | None = None
        with self._lock:
            if not link.alive:
                # Closed while establishing (a late finish after a timeout):
                # forget_link has run, so nothing would ever remove this one.
                superseded = entry
            else:
                held = self._versions.get(link.link_id)
                if held is None or held[0] is not link:
                    held = (link, {})
                    self._versions[link.link_id] = held
                previous = held[1].get(key)
                held[1][key] = entry
                if previous is not None and previous.version is not entry.version:
                    superseded = previous
        if superseded is not None:
            _remove_version(superseded.version)

    def _in_own_loop(self, coroutine: Coroutine[Any, Any, Any], abandoned: threading.Event) -> Any:
        """Run the reader on a fresh event loop in its own thread, off the page loop.

        WHY NOT THE RELAY'S SHARED PAGE LOOP (PR #4563 R1-1). Establish and
        prepare do their file IO synchronously between awaits (a whole-file copy,
        fsyncs, the read-back), so on the page loop one large stat stalled history
        paging for every viewer link. The page loop is shared for the page cache's
        per-loop single-flight, which this op does not use. Nothing here is bound
        to another loop: the AuthStore (and its per-store refresh ``asyncio.Lock``)
        is opened inside this coroutine, and the Radient ``/me`` check opens its
        own ``httpx.AsyncClient`` per call, so a fresh ``asyncio.run`` per request
        is correct.

        WHY A THREAD, NOT ``asyncio.run`` ON THE SLOW-OP WORKER ITSELF. The worker
        must answer the typed ``mentioned_file_owner_timeout`` before the relay's
        overdue timer does, and a blocking ``asyncio.run`` cannot be waited on with
        a timeout. The coroutine owns the auth store it opened, so on a timeout it
        keeps running to its natural end on its own thread and closes the store
        itself, never under running work (R1-3). Cancelling it is not attempted:
        it is inside synchronous file IO and could not be interrupted anyway.
        """
        # R2-3: a request that queued past its budget would be refused at once
        # anyway; starting it would still run the whole audited establishment.
        budget = _wait_budget()
        if budget <= 0:
            coroutine.close()
            raise F._refuse("mentioned_file_owner_timeout")
        # R2-1: bound live readers, abandoned ones included (see _READERS). The
        # same semaphore object is released, even if the module global moves.
        slots = _READERS
        if not slots.acquire(blocking=False):
            coroutine.close()
            raise F._refuse("mentioned_file_owner_busy")
        outcome: concurrent.futures.Future[Any] = concurrent.futures.Future()

        def _late(done: concurrent.futures.Future[Any]) -> None:
            error = done.exception()
            if error is not None:
                logger.debug("%s: abandoned request ended with %r", OP, error)

        def _drive() -> None:
            try:
                try:
                    result = asyncio.run(coroutine)
                finally:
                    # The reader's work, fds and store are finished here, timed
                    # out or not; released before the waiter can see the result
                    # so back-to-back requests never read a stale "busy".
                    slots.release()
            except BaseException as error:  # noqa: BLE001 — handed to the waiter
                outcome.set_exception(error)
            else:
                outcome.set_result(result)

        runner = threading.Thread(target=_drive, name="mesh-mentioned-file", daemon=True)
        try:
            runner.start()
        except BaseException:
            slots.release()
            coroutine.close()
            raise
        try:
            return outcome.result(timeout=budget)
        except concurrent.futures.TimeoutError:
            # The peer is told now; a late establishment must not then hold a
            # version nobody was given (``_serve`` removes it instead).
            abandoned.set()
            outcome.add_done_callback(_late)
            raise F._refuse("mentioned_file_owner_timeout") from None

    def handle(self, link: "PeerLink", frame: dict[str, Any]) -> dict[str, Any]:
        request = _parse(frame)
        entry: _Held | None = None
        if request["file_version"]:
            entry = self._known(link, request["file_version"])
        # _parse guarantees fetch/verify quote a version; restate it for the
        # size checks below rather than trusting that across two functions.
        if request["phase"] != "stat" and entry is None:
            _bad()
        root = F._link_root(link)
        if Path(self.server.root).resolve() != root:
            raise F._refuse("file_approval_listener_required")
        mesh = F._mesh(link, root)
        offset = request["offset"]
        length = 0
        if request["phase"] == "fetch" and entry is not None:
            wanted = request["limit"] if request["limit"] is not None else request["length"]
            wanted = files.FILE_CHUNK_BYTES if wanted is None else wanted
            if offset > entry.version.size:
                _unavailable()
            length = min(wanted, files.FILE_CHUNK_BYTES, entry.version.size - offset)
        elif request["phase"] == "verify" and entry is not None:
            length = request["length"] or 0
            if length <= 0 or length > entry.version.size:
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
        key: _Key = (
            request["session_id"],
            request["provenance_id"],
            request["source_row_id"],
            request["proof_digest"],
        )
        abandoned = threading.Event()
        return self._in_own_loop(
            self._serve(link, root, request, read, key, entry, abandoned), abandoned
        )

    async def _serve(
        self,
        link: "PeerLink",
        root: Path,
        request: dict[str, Any],
        read: B.ReadBinding,
        key: _Key,
        entry: _Held | None,
        abandoned: threading.Event,
    ) -> dict[str, Any]:
        if entry is not None:
            return await self._answer(link, root, request, read, key, entry, abandoned)
        flight = await self._take_flight(link, key, abandoned)
        try:
            return await self._answer(link, root, request, read, key, entry, abandoned)
        finally:
            self._land(link, key, flight)

    async def _answer(
        self,
        link: "PeerLink",
        root: Path,
        request: dict[str, Any],
        read: B.ReadBinding,
        key: _Key,
        entry: _Held | None,
        abandoned: threading.Event,
    ) -> dict[str, Any]:
        reuse = self._held(link, key) if entry is None else None
        # A held version means P1 verified this requester on this link already.
        verified_before = entry is not None or reuse is not None
        # The auth store stays open for the WHOLE request, and is closed by this
        # coroutine only: verify_live reads the native row through it on every
        # await inside establish/prepare.
        with contextlib.ExitStack() as stack:
            try:
                auth = stack.enter_context(F._own_auth(root))
                account = F._owner_serving_account(link, auth, root)
            except MeshRefusal as refusal:
                raise _owner_refusal(refusal, typed=verified_before) from None
            if entry is not None and account.login_incarnation != entry.incarnation:
                raise F._refuse("mentioned_file_owner_relogin")
            relogged = reuse is not None and account.login_incarnation != reuse.incarnation
            if reuse is not None and relogged:
                # A relogin since this was sealed: never reuse it, and stop
                # holding it (it can never prepare again); a fresh
                # establishment re-verifies P1 under the new login.
                self._drop(link, key, reuse)
                reuse = None
            reader = M.OwnerMentionedFileReader(
                incoming_link=link,
                auth=auth,
                owner_account=B.ExactNativeAccountSelection(credential_id=account.credential_id),
                evidence=B.OwnerFileEvidenceReference(association=account.association),
                required_subject=B.CanonicalSubject(
                    issuer=account.issuer, opaque_id=account.opaque_id
                ),
            )
            fresh = False
            try:
                if entry is not None:
                    held, prepared = entry, await reader.prepare(entry.version, read)
                else:
                    held, prepared, fresh = await self._stat(
                        link, key, reader, root, read, reuse, account
                    )
            except MeshRefusal as refusal:
                # A relogin mid-request surfaces inside verify_live as a
                # generic no-active-slot refusal; name it for what it is.
                try:
                    account.require_unchanged(auth, root)
                except MeshRefusal as moved:
                    raise _owner_refusal(moved, typed=verified_before) from None
                if relogged:
                    # S2-5: this requester was verified on this link, and the
                    # login it was verified under is gone; say so, as fetch and
                    # verify of the old version do.
                    raise F._refuse("mentioned_file_owner_relogin") from None
                # S2-2: before P1, P1's own refusals are as revealing as the
                # accessor's (they exist only when exactly one login is usable).
                raise _owner_refusal(refusal, typed=verified_before) from None
            try:
                account.require_unchanged(auth, root)
            except MeshRefusal as moved:
                if fresh:
                    _remove_version(held.version)
                raise _owner_refusal(moved, typed=verified_before) from None
        if fresh:
            if abandoned.is_set():
                _remove_version(held.version)
            else:
                self._install(link, key, held)
        return self._reply(request, prepared, held.name, held.mtime_ns)

    async def _stat(
        self,
        link: "PeerLink",
        key: _Key,
        reader: M.OwnerMentionedFileReader,
        root: Path,
        read: B.ReadBinding,
        reuse: _Held | None,
        account: Any,
    ) -> tuple[_Held, Any, bool]:
        """Reuse the held version while it still prepares, else establish anew.

        ``prepare`` re-runs P1, P2 and the sealed-stamp check and writes its own
        audit row, so a reused version is re-authorized exactly like a re-stat
        that quotes it. Returns ``(held, prepared, fresh)``.
        """
        if reuse is not None:
            again = B.ReadBinding.model_validate(
                {**read.model_dump(), "file_version": reuse.version.commitment}
            )
            try:
                return reuse, await reader.prepare(reuse.version, again), False
            except MeshRefusal:
                # No longer prepares: drop it now, whatever the fresh
                # establishment below then answers (Q2-1).
                self._drop(link, key, reuse)
        initial = M.begin_establishment(read)
        version: Any = None
        try:
            version, prepared = await reader.establish(initial)
            mtime_ns = self._sealed_mtime(version)
            # AFTER establish, whose first step is P1 (S2-1): a proof lookup
            # before it told an unapproved peer whether the named file changed
            # since it was mentioned. A refusal here still strands nothing
            # (S1-4): the cleanup below removes this call's directory.
            name = self._name(root, read)
        except BaseException:
            # A failed establishment leaves a pending (or, when _sealed_mtime
            # or _name refuses, a sealed) directory nothing will ever load. It
            # is this call's own.
            _remove_version_directory(
                root / "sessions" / read.session_id / M._DIRECTORY / initial.nonce,
                initial.nonce,
                None if version is None else version._directory_identity,
            )
            raise
        return _Held(version, name, mtime_ns, account.login_incarnation), prepared, True

    @staticmethod
    def _name(root: Path, read: B.ReadBinding) -> str:
        """Display name from the exact proof this request quotes (bound by its digest)."""
        proof = P.lookup_proof(root, read.session_id, read.provenance_id)
        try:
            if proof is None or "sha256:" + proof["proof_digest"] != read.proof_digest:
                _unavailable()
            return Path(proof["file"]["path"]).name
        except (KeyError, TypeError):
            _unavailable()

    @staticmethod
    def _sealed_mtime(version: Any) -> int:
        """The SEALED source mtime, committed at establishment."""
        try:
            # The seal's source_stamp is file_provenance._identity + st_mode:
            # (dev, ino, ctime_ns, mtime_ns, size, nlink, mode).
            return int(json.loads(version._seal_bytes)["source_stamp"][3])
        except (ValueError, KeyError, IndexError, TypeError):
            _unavailable()

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
