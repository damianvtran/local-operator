"""The session move's WORKING-SET TRANSFER: the mesh half that carries the files.

WHY THIS EXISTS. ``lop sessions move`` used to copy the session directory and nothing
else, so the conversation arrived on the other device without its worktrees, unpushed
commits or dirty files (operator, 2026-11-18: "people could use the move function and
then realize all the work is not available"). PR A (``workset.py``) computes the
manifest and gates the move on the user's confirmation; ``workset_transfer.py`` is the
mesh-independent git engine. THIS module wires the engine into the move protocol of
``mobility.py`` (design ``docs/design/move-working-set.md`` §4, §8, §10):

* **Owner, at prepare (P4)** — :func:`source_prepare` snapshots each confirmed worktree
  into private refs (nothing in the worktree changes), journals the immutable snapshot
  ids under ``<config>/network/workset-out/<sid>/journal.json`` and binds them to the
  confirmed scope digest, the lease epoch and the authenticated destination
  (``workset.transfer_binding``).
* **Owner, ``workset_plan`` (W2)** — :func:`serve_plan` stages one immutable,
  digest-named bundle per worktree (``--not <have>``, every outgoing object scanned
  first) and journals the artifact set. The journal is the ONLY allow-list
  ``workset_fetch`` serves from — never a directory listing.
* **Owner, ``workset_fetch`` (W3/W4)** — :func:`serve_fetch` serves one bounded span of
  one journaled artifact, to the journal's ``to_device`` only, under the journal's
  lease epoch only, after a DURABLE ``session.workset.serve`` audit row for the first
  byte of each artifact and for every resumed (non-contiguous) serve.
* **Destination, between the session copy and its content digest (W1–W5)** —
  :func:`destination_transfer` resolves/clones the primary with ITS OWN credentials,
  asks for the plan, checks the cap and its free space, pulls each artifact into
  ``network/staging/<sid>.workset/<hex>.part`` (resuming only after the owner confirms
  the existing prefix digest), verifies sha256 and ``git bundle verify`` (re-planning
  once with ``have=[]``), and returns the canonical ``workset_digest`` ``ready``
  carries. The owner compares it at commit (:func:`ready_digest_matches`).
* **Destination, after ``done`` and before any engage** — :func:`promote_incoming`
  moves the verified staging into ``network/workset-in/<sid>/`` at the promote, and
  :func:`restore_arrived` restores each worktree. A failure leaves the artifacts and
  ``pending.json``; :func:`resume_pending` retries on relay start.
* **Owner, at commit** — :func:`mark_departed` writes the departure journal + marker.
  THE SOURCE NEVER DELETES A WORKTREE FILE; only this module's own staging
  (``workset-out/<sid>/``) is removed after ``done``/rollback, or by age.

Recall (``--to local``) is the same code with the roles swapped: the remote device is
the owner, this device the destination, and ``restore`` updates a tree in place only
when THIS device's own departure journal recorded it and it re-measures unchanged.

Stdlib only at import time (the relay imports it); the engine imports lazily.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import shutil
import stat
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.network.relay import PeerLink, RelayServer
    from local_operator.network.workset import Manifest

logger = logging.getLogger(__name__)

#: Audit events this module writes (design §8.3). ``serve`` is the required one: it
#: is durable (written and fsynced) BEFORE the first byte of each artifact leaves.
AUDIT_WORKSET_SERVE: Final = "session.workset.serve"
AUDIT_WORKSET_RESTORE: Final = "session.workset.restore"
AUDIT_WORKSET_DEPARTED: Final = "session.workset.departed"

#: Journal / pending schema version.
JOURNAL_VERSION: Final = 1

#: Destination free-space rule (design §8.4): bundle + unpacked objects + checkout.
FREE_SPACE_FACTOR: Final = 3

_HEX64: Final = re.compile(r"^[0-9a-f]{64}$")
_DIGEST: Final = re.compile(r"^sha256:([0-9a-f]{64})$")
_SHA: Final = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")

#: Next contiguous offset served per (session, lease, to_device, digest). A request at
#: exactly that offset continues an audited stream; anything else (offset 0, a gap, a
#: relay restart that forgot the cursor) is a NEW serve and is audited first.
_SERVE_CURSOR: dict[tuple[str, str, str, str], int] = {}
_SERVE_LOCK = threading.Lock()


class WorksetTransferRefused(Exception):
    """A transfer step refused with a move refusal code (``changed: False``)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


# --------------------------------------------------------------------------- paths


def home_of(server: Any) -> Path:
    """The home the working set is remapped under on THIS device (design §7.1).

    ``server.workset_home`` is the seam an install with a non-login home (and the
    two-relay tests, whose relays share one process HOME) sets; absent, the user's
    home. Both the manifest and the restore read it, so they can never disagree.
    """
    explicit = getattr(server, "workset_home", None)
    return Path(explicit) if explicit else Path.home()


def _network(root: Path | str) -> Path:
    return Path(root) / "network"


#: The session-id shape these paths accept — the engine's own rule. Every directory
#: below is later ``rmtree``'d, so an id that could name ``..`` or a separator must
#: never reach a path join here.
_SESSION_ID: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def _sid(session_id: str) -> str:
    if not _SESSION_ID.match(str(session_id)):
        raise ValueError(f"unsafe session id {session_id!r}")
    return str(session_id)


def outgoing_dir(root: Path | str, session_id: str) -> Path:
    """Owner: ``<config>/network/workset-out/<sid>/`` (journal + staged artifacts)."""
    return _network(root) / "workset-out" / _sid(session_id)


def incoming_staging(root: Path | str, session_id: str) -> Path:
    """Destination: ``<config>/network/staging/<sid>.workset/`` until the promote."""
    return _network(root) / "staging" / f"{_sid(session_id)}.workset"


def arrived_dir(root: Path | str, session_id: str) -> Path:
    """Destination: ``<config>/network/workset-in/<sid>/`` after the promote."""
    return _network(root) / "workset-in" / _sid(session_id)


def _private_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink() or not path.is_dir():
        raise WorksetTransferRefused("refused", f"{path} is not a private directory")
    return path


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Same-directory temp + ``os.replace``, 0600: a reader sees the old or the new."""
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=1)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _remove_tree(path: Path) -> None:
    """Remove a directory THIS MODULE created (staging/artifacts), never a symlink."""
    try:
        if path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    except OSError:
        logger.debug("workset: could not remove %s", path, exc_info=True)


# ------------------------------------------------------------- canonical digests


def workset_digest(mapping: Sequence[tuple[str, Sequence[str]]]) -> str:
    """``sha256:`` over the canonical MANIFEST MAPPING ``[(worktree path, [artifact
    digests…]), …]`` in plan order (security lead, 2026-11-18): which bytes belong to
    which worktree, not a sorted multiset of hashes a swap between two worktrees would
    leave unchanged. ``ready`` carries the destination's; the owner recomputes its own
    from its journal at commit."""
    rows = [[str(path), [str(d) for d in digests]] for path, digests in mapping]
    canonical = json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _journal_mapping(journal: Mapping[str, Any]) -> list[tuple[str, list[str]]]:
    artifacts = journal.get("artifacts") or {}
    out: list[tuple[str, list[str]]] = []
    for row in journal.get("worktrees") or []:
        hit = artifacts.get(str(row.get("n")))
        if isinstance(hit, Mapping) and hit.get("digest"):
            out.append((str(row.get("path")), [str(hit["digest"])]))
    return out


# ------------------------------------------------------------------ owner: prepare


def _repo_relative(path: str, worktree: Path, home: Path) -> str | None:
    """A manifest ``excluded`` path (``~/…`` or absolute) as worktree-relative POSIX."""
    expanded = Path(str(home) + path[1:]) if path.startswith("~/") else Path(path)
    try:
        rel = os.path.relpath(os.path.realpath(expanded), os.path.realpath(worktree))
    except ValueError:
        return None
    if rel.startswith("..") or os.path.isabs(rel) or rel == ".":
        return None
    return Path(rel).as_posix()


def _rev(cwd: Path, ref: str) -> str:
    from local_operator.network import workset_transfer as engine

    if not ref:
        return ""
    proc = engine._git(
        cwd, "rev-parse", "--verify", "-q", f"{ref}^{{commit}}", check=False
    )  # noqa: SLF001
    sha = proc.stdout.strip()
    return sha if _SHA.match(sha) else ""


def _config(cwd: Path, key: str) -> str:
    from local_operator.network import workset_transfer as engine

    return engine._git(cwd, "config", "--get", key, check=False).stdout.strip()  # noqa: SLF001


def _pack_bytes(cwd: Path) -> int:
    from local_operator.network import workset_transfer as engine

    out = engine._git(cwd, "count-objects", "-v", check=False).stdout  # noqa: SLF001
    for line in out.splitlines():
        if line.startswith("size-pack:"):
            try:
                return int(line.split(":", 1)[1]) * 1024
            except ValueError:
                return 0
    return 0


def source_prepare(
    server: RelayServer,
    session_id: str,
    manifest: Manifest,
    *,
    lease_epoch: str,
    to_device: str,
    cwd: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """P4 on the owner, after the retire and the P3 re-check: snapshot and journal.

    Returns ``(handoff_block, offer)``: ``handoff_block`` goes into the handoff journal
    entry (the binding the commit trusts), ``offer`` into the prepare reply (what the
    destination needs for W1–W5). A worktree whose snapshot fails is reported
    ``left_behind`` with the engine's reason; it never fails the move (the source
    keeps it untouched either way).
    """
    from local_operator.network import workset
    from local_operator.network import workset_transfer as engine

    home = home_of(server)
    out = _private_dir(outgoing_dir(server.root, session_id))
    rows: list[dict[str, Any]] = []
    offer_rows: list[dict[str, Any]] = []
    left: list[dict[str, Any]] = []
    for n, facts in enumerate(manifest.planned):
        worktree = Path(facts.abs_path)
        excludes = [
            rel
            for rel in (_repo_relative(item.path, worktree, home) for item in facts.excluded)
            if rel
        ]
        try:
            snap = engine.snapshot(worktree, session_id, n, excludes)
        except engine.MoveWorksetError as exc:
            left.append({"path": facts.path, "reason": exc.reason})
            continue
        upstream_merge = _config(worktree, f"branch.{snap.branch}.merge") if snap.branch else ""
        tips = [
            t
            for t in dict.fromkeys(
                [snap.head, _rev(worktree, facts.upstream), _rev(worktree, facts.base)]
            )
            if t
        ]
        row = {
            "n": n,
            "path": facts.path,
            "repo": facts.repo,
            "abs_path": str(worktree),
            "remote_url": facts.remote_url,
            "is_primary": bool(facts.is_primary),
            "snapshot": asdict(snap),
            "tips": tips,
        }
        rows.append(row)
        offer_rows.append(
            {
                "n": n,
                "path": facts.path,
                "repo": facts.repo,
                "remote_url": facts.remote_url,
                "branch": snap.branch,
                "upstream_merge": upstream_merge or None,
                "head": snap.head,
                "index": snap.index,
                "worktree": snap.worktree,
                "is_primary": bool(facts.is_primary),
                "pack_bytes": _pack_bytes(worktree),
                "tips": tips,
            }
        )
    scope = workset.scope_digest(manifest)
    binding = workset.transfer_binding(
        scope,
        lease_epoch=lease_epoch,
        to_device=to_device,
        snapshots=[
            {"path": r["path"], **{k: r["snapshot"][k] for k in ("head", "index", "worktree")}}
            for r in rows
        ],
    )
    journal = {
        "version": JOURNAL_VERSION,
        "session_id": session_id,
        "lease_epoch": lease_epoch,
        "to_device": to_device,
        "scope_digest": scope,
        "binding": binding,
        "cap_bytes": int(manifest.cap_bytes),
        "worktrees": rows,
        "left_behind": left,
        "artifacts": {},
    }
    _write_json(out / "journal.json", journal)
    handoff_block = {"binding": binding, "scope_digest": scope, "worktrees": len(rows)}
    offer = {
        "version": workset.WORKSET_TRANSFER_VERSION,
        # The session's cwd in ``~/`` form, so the destination can point the arrived
        # session at the RESTORED tree (the promote's cwd mapping runs before the
        # restore exists). Empty when it is outside this device's home.
        "cwd": engine.home_relative(cwd, home=home) or "" if cwd else "",
        "binding": binding,
        "worktrees": offer_rows,
        "left_behind": left,
        "cap_bytes": int(manifest.cap_bytes),
    }
    return handoff_block, offer


# --------------------------------------------------------- owner: authorisation


def _refused(code: str, message: str, session_id: str) -> dict[str, Any]:
    return {"result": "refused", "code": code, "message": message, "session_id": session_id}


def _authorised_journal(
    server: RelayServer, link: PeerLink, frame: Mapping[str, Any], session_id: str
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """``(journal, None)`` when this request belongs to the CURRENT move, else
    ``(None, refusal)``.

    THE BINDING CHECK every ``workset_*`` transfer phase runs (design §8.2): the
    in-flight handoff entry is this device's SOURCE ``prepared`` entry; the caller is
    the entry's ``to_device`` (the AUTHENTICATED link peer, never a frame field); the
    frame's lease epoch equals the entry's; the journal on disk is the one that entry
    bound (same epoch, same destination, same binding digest). Any mismatch is a
    refusal with zero bytes served.
    """
    from local_operator.session.placement import (
        HANDOFF_PHASE_PREPARED,
        handoff_in_flight,
    )

    try:
        entry = handoff_in_flight(Path(server.root), session_id)
    except Exception:  # noqa: BLE001 — an unreadable journal authorises nothing
        entry = None
    if (
        not entry
        or str(entry.get("role") or "source") != "source"
        or str(entry.get("mode") or "") != "move"
        or str(entry.get("phase") or "") != HANDOFF_PHASE_PREPARED
    ):
        return None, _refused(
            "not_owner", "no move of this conversation is being prepared here", session_id
        )
    if str(entry.get("to_device") or "") != str(link.device_id):
        return None, _refused("not_authorised", "this move is not going to that device", session_id)
    epoch = str(frame.get("lease_epoch") or "")
    if not epoch or epoch != str(entry.get("lease_epoch") or ""):
        return None, _refused(
            "refused", "that request names a different move (lease epoch)", session_id
        )
    bound = entry.get("working_set") if isinstance(entry.get("working_set"), Mapping) else {}
    journal = _read_json(outgoing_dir(server.root, session_id) / "journal.json")
    if (
        journal is None
        or str(journal.get("lease_epoch") or "") != epoch
        or str(journal.get("to_device") or "") != str(link.device_id)
        or not bound
        or str(journal.get("binding") or "") != str(bound.get("binding") or "")
    ):
        return None, _refused(
            "refused", "this move carries no working set for that device", session_id
        )
    return journal, None


# ---------------------------------------------------------------- owner: W2 plan


def serve_plan(server: RelayServer, link: PeerLink, frame: Mapping[str, Any]) -> dict[str, Any]:
    """``net_session_move {phase: "workset_plan", session_id, lease_epoch, have}``.

    Stages one bundle per journaled worktree (``refs/lop/move/<sid>/<n>/* --not
    <have[n]>``) and journals ``n -> {digest, bytes}``. A worktree whose outgoing
    objects the scan flags (or cannot read) is ``left_behind`` with the reason — there
    is no override (manager, 2026-11-18). Over the cap → ``working_set_over_cap``.
    """
    from local_operator.network import workset_transfer as engine

    session_id = str(frame.get("session_id") or "")
    journal, refusal = _authorised_journal(server, link, frame, session_id)
    if refusal is not None or journal is None:
        return refusal or _refused("refused", "no working set", session_id)
    have_value = frame.get("have")
    raw_have: Mapping[str, Any] = have_value if isinstance(have_value, Mapping) else {}
    out = outgoing_dir(server.root, session_id)
    artifacts: dict[str, dict[str, Any]] = {}
    reply_artifacts: list[dict[str, Any]] = []
    left: list[dict[str, Any]] = []
    # The destination DECLINES a worktree whose repository it neither has nor could
    # clone (reason from W1): bundling it would ship the whole history for a restore
    # that cannot happen, so it stays here, reported with the destination's reason.
    decline_value = frame.get("decline")
    raw_decline: Mapping[str, Any] = decline_value if isinstance(decline_value, Mapping) else {}
    for row in journal.get("worktrees") or []:
        n = str(row.get("n"))
        if n in raw_decline:
            reason = str(raw_decline.get(n) or "destination_cannot_clone")
            if not re.fullmatch(r"[a-z_]{1,40}", reason):
                reason = "destination_cannot_clone"
            left.append({"n": int(row.get("n") or 0), "path": row.get("path"), "reason": reason})
            continue
        have = [str(h) for h in (raw_have.get(n) or []) if _SHA.match(str(h))][:64]
        refs = list(engine.SnapshotRefs(**row["snapshot"]).refs)
        repo = Path(str(row.get("abs_path")))
        try:
            try:
                artifact = engine.stage_bundle(repo, refs, have, out)
            except engine.MoveWorksetError as exc:
                if exc.reason != "empty_bundle" or not have:
                    raise
                # The destination already holds every outgoing object (an earlier
                # move of the same tree): a full bundle still carries the refs the
                # restore fetches, and is scanned the same way.
                artifact = engine.stage_bundle(repo, refs, [], out)
        except engine.HistoryWithheld as withheld:
            left.append(
                {
                    "n": int(row.get("n") or 0),
                    "path": row.get("path"),
                    "reason": (
                        "history_unscannable"
                        if all(f.why == "history_unscannable" for f in withheld.check.findings)
                        else "history_secret"
                    ),
                    "excluded": [f.to_dict() for f in withheld.check.findings][:50],
                }
            )
            continue
        except engine.MoveWorksetError as exc:
            left.append(
                {"n": int(row.get("n") or 0), "path": row.get("path"), "reason": exc.reason}
            )
            continue
        artifacts[n] = {"digest": artifact.digest, "bytes": artifact.bytes}
        reply_artifacts.append(
            {"n": int(row.get("n") or 0), "digest": artifact.digest, "bytes": artifact.bytes}
        )
    total = sum(int(a["bytes"]) for a in artifacts.values())
    cap = int(journal.get("cap_bytes") or 0)
    if cap and total > cap:
        journal["artifacts"] = {}
        _write_json(out / "journal.json", journal)
        return _refused(
            "working_set_over_cap",
            f"the working files are {total} bytes once bundled, over this device's "
            f"cap of {cap} (mesh.move.working_set_max_bytes), so nothing was moved",
            session_id,
        )
    journal["artifacts"] = artifacts
    journal["plan_left_behind"] = left
    _write_json(out / "journal.json", journal)
    return {
        "result": "planned",
        "session_id": session_id,
        "artifacts": reply_artifacts,
        "left_behind": left,
    }


# --------------------------------------------------------- owner: W3 audit, W4 fetch


def _audit_workset_serve(
    server: RelayServer,
    *,
    session_id: str,
    to_device: str,
    lease_epoch: str,
    digest: str,
    size: int,
    offset: int,
    path: str,
) -> bool:
    """Write ONE durable ``session.workset.serve`` row; ``True`` only if it is on disk.

    WHY NOT ``AuditLog.record``: it never raises and only flushes on a timer, so it
    cannot gate bytes (design §8.3). PR #4228's ``AuditLog.record_required`` is the
    intended API and was NOT merged when this landed (checked 2026-11-18,
    ``gh pr view 4228``: OPEN), so this is the stand-in, kept behind this one function
    so the swap is a one-line change. It is durable on the same terms as the log's own
    DURABLE_EVENTS: the event is listed there, so ``record`` flushes AND fsyncs it
    before returning; this function holds the writer's lock across the call and then
    requires that (a) the row's sequence number is published to the file and (b) the
    fsync counter moved — the two facts ``flush``/``sync`` only change on success.
    Any doubt is ``False``, and the caller serves zero bytes. Metadata is bounded: no
    payload, no secret, no file content — ids, sizes and one home-relative path.
    """
    from local_operator.network.audit import AuditEvent

    log = getattr(server, "audit", None)
    if log is None or not getattr(log, "_enabled", False):
        return False
    try:
        lock = log._write_lock  # noqa: SLF001 — one writer; see the docstring
        with lock:
            syncs = int(getattr(log, "syncs", 0))
            log.record(
                AuditEvent(
                    event=AUDIT_WORKSET_SERVE,
                    actor=server.identity.device_id,
                    subject=session_id,
                    outcome="ok",
                    detail={
                        "peer": to_device,
                        "to_device": to_device,
                        "lease_epoch": lease_epoch,
                        "artifact": digest,
                        "bytes": int(size),
                        "offset": int(offset),
                        "path": path[:512],
                    },
                )
            )
            seq = int(log.recorded_through)
            return log.publication_of(seq) == "file" and int(getattr(log, "syncs", 0)) > syncs
    except Exception:  # noqa: BLE001 — an audit that cannot be proven is a refusal
        logger.warning("workset: could not write the serve audit for %s", session_id, exc_info=True)
        return False


def _open_artifact(out: Path, hexdigest: str) -> tuple[int, int]:
    """``(fd, size)`` of a staged artifact, opened ``O_NOFOLLOW`` and checked regular."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    if out.is_symlink():
        raise OSError("staging directory is a symlink")
    fd = os.open(out / hexdigest, flags)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        os.close(fd)
        raise OSError("not a regular file")
    return fd, info.st_size


def serve_fetch(server: RelayServer, link: PeerLink, frame: Mapping[str, Any]) -> dict[str, Any]:
    """``net_session_move {phase: "workset_fetch", session_id, lease_epoch, digest,
    offset, limit}`` → one span, or ``{prefix}`` → the sha256 of the first N bytes.

    Authorised per request against the CURRENT move's journaled artifact set (never a
    directory listing: the directory also holds ``<hex>.json`` sidecars and the
    journal). Only a ``sha256:<64 hex>`` name is ever opened. The prefix form serves
    no content; it lets the destination prove its ``.part`` before resuming.
    """
    from local_operator.network import sync as sync_mod

    session_id = str(frame.get("session_id") or "")
    journal, refusal = _authorised_journal(server, link, frame, session_id)
    if refusal is not None or journal is None:
        return refusal or _refused("refused", "no working set", session_id)
    match = _DIGEST.match(str(frame.get("digest") or ""))
    staged = {str(a.get("digest")): n for n, a in (journal.get("artifacts") or {}).items()}
    if match is None or str(frame.get("digest")) not in staged:
        return _refused("unknown_item", "that artifact is not part of this move", session_id)
    digest = str(frame["digest"])
    hexdigest = match.group(1)
    try:
        offset = int(frame.get("offset") or 0)
        limit = int(frame.get("limit") or sync_mod.SYNC_CHUNK_BYTES)
        prefix = frame.get("prefix")
        prefix_len = int(prefix) if prefix is not None else None
    except (TypeError, ValueError):
        return _refused("bad_request", "offset/limit must be integers", session_id)
    if offset < 0 or limit <= 0 or (prefix_len is not None and prefix_len < 0):
        return _refused("bad_request", "offset/limit out of range", session_id)
    limit = min(limit, sync_mod.SYNC_CHUNK_BYTES)
    out = outgoing_dir(server.root, session_id)
    try:
        fd, size = _open_artifact(out, hexdigest)
    except OSError:
        return _refused("unknown_item", "that artifact is no longer staged here", session_id)
    try:
        with os.fdopen(fd, "rb", closefd=True) as handle:
            if prefix_len is not None:
                if prefix_len > size:
                    return _refused("bad_request", "prefix is longer than the artifact", session_id)
                h = hashlib.sha256()
                remaining = prefix_len
                while remaining:
                    block = handle.read(min(remaining, 1 << 20))
                    if not block:
                        break
                    h.update(block)
                    remaining -= len(block)
                return {
                    "result": "prefix",
                    "session_id": session_id,
                    "digest": digest,
                    "prefix": prefix_len,
                    "prefix_sha256": h.hexdigest(),
                    "size": size,
                }
            if offset > size:
                return _refused("bad_request", "offset is past the end of the artifact", session_id)
            epoch = str(journal.get("lease_epoch") or "")
            key = (session_id, epoch, str(link.device_id), digest)
            with _SERVE_LOCK:
                continuing = offset > 0 and _SERVE_CURSOR.get(key) == offset
            if not continuing:
                row_path = next(
                    (
                        str(r.get("path") or "")
                        for r in journal.get("worktrees") or []
                        if str(r.get("n")) == staged[digest]
                    ),
                    "",
                )
                if not _audit_workset_serve(
                    server,
                    session_id=session_id,
                    to_device=str(link.device_id),
                    lease_epoch=epoch,
                    digest=digest,
                    size=size,
                    offset=offset,
                    path=row_path,
                ):
                    return _refused(
                        "audit_unavailable",
                        "this device could not record the transfer in its audit log, so "
                        "no working file was sent",
                        session_id,
                    )
            span, total, eof = sync_mod.read_open_span(
                handle, offset=offset, limit=limit, size=size
            )
    except OSError:
        return _refused("unknown_item", "that artifact could not be read", session_id)
    with _SERVE_LOCK:
        _SERVE_CURSOR[key] = offset + len(span)
        if eof:
            _SERVE_CURSOR.pop(key, None)
    return {
        "result": "chunk",
        "session_id": session_id,
        "digest": digest,
        "offset": offset,
        "size": total,
        "eof": bool(eof),
        "data": base64.b64encode(span).decode("ascii"),
        "chunk_sha256": hashlib.sha256(span).hexdigest(),
    }


# ------------------------------------------------------------- owner: commit/done


def ready_digest_matches(
    root: Path | str, session_id: str, entry: Mapping[str, Any], reported: str
) -> bool:
    """Commit check (design §4 step 12): the destination's ``workset_digest`` equals the
    owner's, recomputed from ITS journal. A move without a bound working set must not
    report one (an empty reported value is the only match)."""
    bound = entry.get("working_set") if isinstance(entry.get("working_set"), Mapping) else None
    if not bound:
        return not reported
    journal = _read_json(outgoing_dir(root, session_id) / "journal.json")
    if journal is None or str(journal.get("binding") or "") != str(bound.get("binding") or ""):
        return False
    return reported == workset_digest(_journal_mapping(journal))


def mark_departed(server: RelayServer, session_id: str, to_device: str) -> list[dict[str, Any]]:
    """At the source's commit: departure journal + ``lop-moved.json`` marker per
    transferred worktree (design §8.6). NEVER a delete; best effort — a marker that
    cannot be written must not undo a committed move."""
    from local_operator.network import workset_transfer as engine

    journal = _read_json(outgoing_dir(server.root, session_id) / "journal.json")
    if not journal:
        return []
    sent = set((journal.get("artifacts") or {}).keys())
    snaps = [
        engine.SnapshotRefs(**row["snapshot"])
        for row in journal.get("worktrees") or []
        if str(row.get("n")) in sent
    ]
    if not snaps:
        return []
    try:
        name = str(server._member_name(to_device) or "")  # noqa: SLF001
    except Exception:  # noqa: BLE001
        name = ""
    try:
        marked = engine.mark_departed(
            Path(server.root), session_id, to_device, snaps, name=name, home=home_of(server)
        )
    except Exception:  # noqa: BLE001 — a marker is not a gate
        logger.warning("workset: could not mark %s's worktrees departed", session_id, exc_info=True)
        return []
    _plain_audit(server, AUDIT_WORKSET_DEPARTED, session_id, to_device, marked=len(marked))
    return marked


def discard_outgoing(root: Path | str, session_id: str) -> None:
    """Remove ``workset-out/<sid>/`` (after ``done``, or on rollback). The snapshot refs
    in the source repository are KEPT: they are the restore point (design §8.6)."""
    try:
        _remove_tree(outgoing_dir(root, session_id))
    except ValueError:
        logger.error("workset: refusing to sweep artifacts for an unsafe id")


def _plain_audit(server: Any, event: str, session_id: str, peer: str, **detail: Any) -> None:
    try:
        from local_operator.network.audit import AuditEvent

        server.audit.record(
            AuditEvent(
                event=event,
                actor=server.identity.device_id,
                subject=session_id,
                outcome="ok",
                detail={"peer": peer, **detail},
            )
        )
    except Exception:  # noqa: BLE001 — a record is not a gate
        logger.debug("workset: could not record %s", event, exc_info=True)


# ------------------------------------------------------------ destination: W1–W5


def _ask(transport: Any, frame: dict[str, Any]) -> dict[str, Any]:
    from local_operator.network.mobility import Moved

    try:
        return transport.ask(frame)
    except Moved as refusal:
        code = (
            refusal.code
            if refusal.code in ("working_set_over_cap",)
            else "working_set_transfer_failed"
        )
        raise WorksetTransferRefused(code, refusal.message) from refusal


def _fetch_artifact(
    transport: Any,
    *,
    session_id: str,
    lease_epoch: str,
    digest: str,
    expected_bytes: int,
    into: Path,
) -> Path:
    """W4: pull one artifact into ``into/<hex>.part``, verify, land as ``into/<hex>``.

    Resumes from an existing ``.part`` ONLY after the owner confirms its prefix digest;
    a corrupted prefix is discarded and the pull restarts at 0. Every chunk is checked
    against the offset asked for, its own sha256 and the artifact size; the landed file
    must hash to ``digest`` or nothing lands.
    """
    from local_operator.network import sync as sync_mod

    hexdigest = digest.removeprefix("sha256:")
    final = into / hexdigest
    if final.is_file() and not final.is_symlink() and _sha256(final) == hexdigest:
        return final
    part = into / f"{hexdigest}.part"
    base = {
        "op": "net_session_move",
        "phase": "workset_fetch",
        "session_id": session_id,
        "lease_epoch": lease_epoch,
        "digest": digest,
    }
    offset = 0
    if part.is_file() and not part.is_symlink():
        have = part.stat().st_size
        if 0 < have <= expected_bytes:
            local = _sha256(part)
            answer = _ask(transport, {**base, "prefix": have})
            if str(answer.get("prefix_sha256") or "") == local:
                offset = have
        if offset == 0:
            part.unlink()
    flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(part, flags, 0o600)
    with os.fdopen(fd, "r+b") as handle:
        handle.truncate(offset)
        handle.seek(offset)
        while True:
            answer = _ask(transport, {**base, "offset": offset, "limit": sync_mod.SYNC_CHUNK_BYTES})
            if str(answer.get("digest") or "") != digest or int(answer.get("offset", -1)) != offset:
                raise WorksetTransferRefused(
                    "working_set_transfer_failed", "the owner answered a different span"
                )
            data = base64.b64decode(str(answer.get("data") or ""), validate=True)
            if hashlib.sha256(data).hexdigest() != str(answer.get("chunk_sha256") or ""):
                raise WorksetTransferRefused(
                    "working_set_transfer_failed", "a chunk failed its digest"
                )
            size = int(answer.get("size") or 0)
            if size != expected_bytes or offset + len(data) > size:
                raise WorksetTransferRefused(
                    "working_set_transfer_failed", "the artifact size changed"
                )
            handle.write(data)
            offset += len(data)
            if answer.get("eof") or offset >= size:
                break
            if not data:
                raise WorksetTransferRefused(
                    "working_set_transfer_failed", "the owner sent an empty chunk"
                )
        handle.flush()
        os.fsync(handle.fileno())
    if _sha256(part) != hexdigest:
        part.unlink(missing_ok=True)
        raise WorksetTransferRefused("working_set_transfer_failed", "an artifact failed its sha256")
    os.replace(part, final)
    return final


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _plan(
    transport: Any,
    session_id: str,
    lease_epoch: str,
    have: Mapping[str, list[str]],
    decline: Mapping[str, str],
) -> dict[str, Any]:
    return _ask(
        transport,
        {
            "op": "net_session_move",
            "phase": "workset_plan",
            "session_id": session_id,
            "lease_epoch": lease_epoch,
            "have": dict(have),
            "decline": dict(decline),
        },
    )


def destination_transfer(
    server: RelayServer,
    transport: Any,
    session_id: str,
    *,
    lease_epoch: str,
    offer: Mapping[str, Any],
    owner_device: str,
) -> str:
    """W1–W5 on the destination. Returns the ``workset_digest`` for ``ready``.

    Raises :class:`WorksetTransferRefused` (the caller rolls the move back exactly as a
    failed session copy: nothing committed, ``changed: False``). Writes
    ``staging/<sid>.workset/pending.json`` with one restore plan per artifact, for the
    restore that runs after ``done``.
    """
    from local_operator.network import workset
    from local_operator.network import workset_transfer as engine

    home = home_of(server)
    into = _private_dir(incoming_staging(server.root, session_id))
    rows = [r for r in offer.get("worktrees") or [] if isinstance(r, Mapping)]
    rewrite_notes: list[str] = []
    have: dict[str, list[str]] = {}
    decline: dict[str, str] = {}
    primaries: dict[str, dict[str, Any]] = {}
    for row in rows:
        n = str(row.get("n"))
        remote_url = str(row.get("remote_url") or "")
        repo_rel = str(row.get("repo") or "")
        use_url = remote_url
        res = engine.resolve_primary(use_url, repo_rel, home=home) if use_url else None
        cloned = False
        if res is not None and res.state == "absent":
            # THE HTTPS REWRITE (the destination probe's offer, design §4 W1): an ssh
            # remote this device has no key for is cloned over its https form instead,
            # with THIS device's own credentials. Tried in order — the original URL,
            # then the rewrite — so a destination that can use ssh keeps it. The
            # rewrite is repo-local by construction (it is the fresh clone's origin;
            # a checkout the user already had is never re-pointed) and is named in
            # the receipt's notes. ``https_rewrite.url`` on the row is the probe's
            # offer when an owner forwards it; else this device derives the same form.
            from local_operator.network import workset_probe

            offered = row.get("https_rewrite")
            alt = (
                str(offered.get("url") or "")
                if isinstance(offered, Mapping)
                else workset_probe.https_rewrite(remote_url)
            )
            candidates = [remote_url] + ([alt] if alt and alt != remote_url else [])
            res = engine.PrimaryResolution("absent", None, reason="destination_cannot_clone")
            for candidate in candidates:
                try:
                    got = engine.clone_primary(
                        candidate,
                        repo_rel,
                        home=home,
                        pack_bytes=int(row.get("pack_bytes") or 0),
                    )
                except engine.MoveWorksetError as exc:
                    if exc.reason == "insufficient_space":
                        raise WorksetTransferRefused(
                            "insufficient_space",
                            f"this device does not have room to clone {repo_rel} "
                            f"({exc.detail}), so nothing was moved",
                        ) from exc
                    continue
                if got.state == "present":
                    res, use_url, cloned = got, candidate, True
                    if candidate != remote_url:
                        rewrite_notes.append(f"{repo_rel}: {remote_url} -> {candidate}")
                    break
        primaries[n] = {"res": res, "url": use_url, "cloned": cloned}
        if res is not None and res.state == "present" and res.path is not None:
            tips = [str(t) for t in row.get("tips") or [] if _SHA.match(str(t))]
            have[n] = engine.compute_have(res.path, tips, remote=res.remote)
        else:
            decline[n] = (res.reason if res is not None else "") or "destination_cannot_clone"
    planned = _plan(transport, session_id, lease_epoch, have, decline)
    artifacts = {
        str(a.get("n")): a for a in planned.get("artifacts") or [] if isinstance(a, Mapping)
    }
    left = [dict(r) for r in planned.get("left_behind") or [] if isinstance(r, Mapping)]
    total = sum(int(a.get("bytes") or 0) for a in artifacts.values())
    cap = workset.cap_bytes(Path(server.root))
    if total > cap:
        raise WorksetTransferRefused(
            "working_set_over_cap",
            f"the working files are {total} bytes, over this device's cap of {cap} "
            "(mesh.move.working_set_max_bytes), so nothing was moved",
        )
    free = shutil.disk_usage(into).free
    if total * FREE_SPACE_FACTOR > free:
        raise WorksetTransferRefused(
            "insufficient_space",
            f"this device has {free} bytes free and the working files need about "
            f"{total * FREE_SPACE_FACTOR}, so nothing was moved",
        )
    plans: list[dict[str, Any]] = []
    mapping: list[tuple[str, list[str]]] = []
    for row in rows:
        n = str(row.get("n"))
        art = artifacts.get(n)
        if art is None:
            continue
        digest = str(art.get("digest") or "")
        if not _DIGEST.match(digest):
            raise WorksetTransferRefused(
                "working_set_transfer_failed", "the owner named a malformed artifact"
            )
        info = primaries.get(n) or {}
        res = info.get("res")
        fetched = _fetch_artifact(
            transport,
            session_id=session_id,
            lease_epoch=lease_epoch,
            digest=digest,
            expected_bytes=int(art.get("bytes") or 0),
            into=into,
        )
        if res is not None and res.state == "present" and res.path is not None:
            if not engine.verify_bundle(res.path, fetched):
                # W5: re-plan ONCE with ``have=[]`` for this worktree; a second failure
                # rolls the whole move back (nothing committed anywhere).
                replanned = _plan(transport, session_id, lease_epoch, {**have, n: []}, decline)
                again = {str(a.get("n")): a for a in replanned.get("artifacts") or []}.get(n)
                if not again:
                    raise WorksetTransferRefused(
                        "working_set_transfer_failed", "the re-plan carried nothing"
                    )
                digest = str(again.get("digest") or "")
                if not _DIGEST.match(digest):
                    raise WorksetTransferRefused(
                        "working_set_transfer_failed", "the owner named a malformed artifact"
                    )
                fetched = _fetch_artifact(
                    transport,
                    session_id=session_id,
                    lease_epoch=lease_epoch,
                    digest=digest,
                    expected_bytes=int(again.get("bytes") or 0),
                    into=into,
                )
                if not engine.verify_bundle(res.path, fetched):
                    raise WorksetTransferRefused(
                        "working_set_transfer_failed",
                        f"the bundle for {row.get('path')} did not verify here, even in full",
                    )
                # Every other worktree's artifact is unchanged by a re-plan (the
                # staged directory is the cache), so their digests still hold.
                artifacts = {
                    **artifacts,
                    **{str(a.get("n")): a for a in replanned.get("artifacts") or []},
                }
        mapping.append((str(row.get("path")), [digest]))
        plans.append(
            {
                "session_id": session_id,
                "n": int(row.get("n") or 0),
                "bundle_digest": digest,
                "path": str(row.get("path") or ""),
                "primary": str(row.get("repo") or ""),
                "remote_url": str(info.get("url") or row.get("remote_url") or ""),
                "head": str(row.get("head") or ""),
                "index": str(row.get("index") or ""),
                "worktree": str(row.get("worktree") or ""),
                "branch": row.get("branch") or None,
                "upstream_merge": row.get("upstream_merge") or None,
                "src_device": owner_device,
                "is_primary": bool(row.get("is_primary")),
                "fresh_clone": bool(info.get("cloned")),
            }
        )
    digest_all = workset_digest(mapping)
    _write_json(
        into / "pending.json",
        {
            "version": JOURNAL_VERSION,
            "session_id": session_id,
            "from_device": owner_device,
            "lease_epoch": lease_epoch,
            "workset_digest": digest_all,
            "plans": plans,
            "left_behind": [*[dict(r) for r in offer.get("left_behind") or []], *left],
            "notes": rewrite_notes,
            "cwd": str(offer.get("cwd") or ""),
            "restored": {},
        },
    )
    return digest_all


def discard_incoming(root: Path | str, session_id: str) -> None:
    """The destination rollback's half: remove ``staging/<sid>.workset/``."""
    try:
        _remove_tree(incoming_staging(root, session_id))
    except ValueError:
        logger.error("workset: refusing to sweep staging for an unsafe id")


def promote_incoming(root: Path | str, session_id: str) -> bool:
    """At the session promote: ``staging/<sid>.workset`` → ``workset-in/<sid>``.

    An older ``workset-in/<sid>`` (an earlier round trip of the same session) is this
    module's own cache of artifacts and is replaced. Returns whether a working set
    arrived.
    """
    src = incoming_staging(root, session_id)
    if not src.is_dir() or src.is_symlink() or not (src / "pending.json").is_file():
        return False
    dst = arrived_dir(root, session_id)
    _private_dir(dst.parent)
    _remove_tree(dst)
    os.replace(src, dst)
    return True


# --------------------------------------------------------- destination: restore


def restore_arrived(server: Any, session_id: str) -> dict[str, Any] | None:
    """Restore every pending worktree of ``session_id`` (after ``done``, before engage).

    Idempotent: each finished worktree is recorded in ``pending.json`` and skipped on a
    retry. Returns the receipt fragment ``{state, worktrees:[…], notes}`` or ``None``
    when no working set arrived. A per-worktree failure becomes ``pending_restore``
    with its reason (the source still holds the untouched tree, so nothing is lost).
    """
    from local_operator.network import workset_transfer as engine

    directory = arrived_dir(server.root, session_id)
    pending = _read_json(directory / "pending.json")
    if pending is None:
        return None
    home = home_of(server)
    restored: dict[str, Any] = dict(pending.get("restored") or {})
    for raw in pending.get("plans") or []:
        key = str(raw.get("n"))
        if key in restored and restored[key].get("state") != "pending_restore":
            continue
        plan = engine.RestorePlan(**raw)
        try:
            report = engine.restore(directory, plan, home=home, config=Path(server.root))
        except engine.MoveWorksetError as exc:
            report = engine.RestoreReport("pending_restore", plan.path, reason=exc.reason)
        except Exception as exc:  # noqa: BLE001 — a restore failure is reported, not raised
            logger.warning("workset: restore of %s failed", plan.path, exc_info=True)
            report = engine.RestoreReport("pending_restore", plan.path, reason=type(exc).__name__)
        row = asdict(report)
        if report.state == "restored" and report.dest_path:
            try:
                dest = engine.contained_path(home, report.dest_path)
                primary = (
                    engine.contained_path(home, plan.primary)
                    if plan.primary.startswith("~/")
                    else dest
                )
                row["recipes"] = [r.to_dict() for r in engine.recipes_for(dest, primary, home=home)]
            except Exception:  # noqa: BLE001 — a recipe is advice
                row["recipes"] = []
        restored[key] = row
        pending["restored"] = restored
        _write_json(directory / "pending.json", pending)
    rows = [
        restored[str(p.get("n"))] for p in pending.get("plans") or [] if str(p.get("n")) in restored
    ]
    _retarget_cwd(server, session_id, str(pending.get("cwd") or ""), rows, home)
    waiting = [r for r in rows if r.get("state") == "pending_restore"]
    state = "pending_restore" if waiting else "restored"
    if not waiting:
        # Every artifact is in a repository now; drop the bundles, keep the record.
        for child in directory.iterdir():
            if _HEX64.match(child.name) or child.name.endswith(".part"):
                child.unlink(missing_ok=True)
    _plain_audit(
        server,
        AUDIT_WORKSET_RESTORE,
        session_id,
        str(pending.get("from_device") or ""),
        state=state,
        worktrees=[{"state": r.get("state"), "reason": r.get("reason") or ""} for r in rows][:50],
    )
    return {
        "state": state,
        "worktrees": rows,
        "left_behind": list(pending.get("left_behind") or []),
        "notes": list(pending.get("notes") or []),
    }


def _retarget_cwd(
    server: Any, session_id: str, cwd_rel: str, rows: Sequence[Mapping[str, Any]], home: Path
) -> None:
    """Point the arrived session's cwd marker at the tree that now holds its work.

    The promote maps the recorded cwd (``carry.rewrite_host_cwd``) BEFORE the restore
    has created the worktree, so it cannot pick the restored path; and a restore that
    landed ``beside`` (conflict suffix) or under ``.worktrees/moved-<sid8>`` is somewhere
    no path rule could guess. The source's ``~/`` cwd inside a restored row's source
    path is rewritten to the same tail under that row's ``dest_path``, contained under
    this device's home. Best effort: a marker it cannot write keeps the promote's value.
    """
    from local_operator.network import carry
    from local_operator.network import workset_transfer as engine

    if not cwd_rel.startswith("~/"):
        return
    for row in rows:
        src, dest = str(row.get("path") or ""), str(row.get("dest_path") or "")
        if row.get("state") != "restored" or not src.startswith("~/") or not dest:
            continue
        if cwd_rel != src and not cwd_rel.startswith(src.rstrip("/") + "/"):
            continue
        try:
            target = engine.contained_path(home, dest + cwd_rel[len(src) :])
            if not target.is_dir():
                target = engine.contained_path(home, dest)
            marker_path = Path(server.root) / "sessions" / session_id / carry.DESKTOP_MARKER_NAME
            marker = _read_json(marker_path) or {}
            marker["cwd"] = str(target)
            carry._write_marker(marker_path, marker)  # noqa: SLF001 — the one marker writer
        except Exception:  # noqa: BLE001 — advice for the next open, never a failure
            logger.debug("workset: could not retarget %s's cwd", session_id, exc_info=True)
        return


def arrival_receipt(
    block: Mapping[str, Any] | None, restored: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """The receipt's ``working_set`` block once the transfer ran (destination side).

    Starts from the owner's arrival block (the confirmed scope) and replaces each
    planned row's state with what happened HERE: ``restored`` (with ``dest_path``,
    ``branch``, ``mode`` and any recipes), ``pending_restore`` or ``left_behind`` with
    the reason. ``transferred`` is true once artifacts arrived.
    """
    if block is None:
        return None
    out = dict(block)
    if restored is None:
        return out
    by_path = {str(r.get("path")): r for r in restored.get("worktrees") or []}
    left = {str(r.get("path")): r for r in restored.get("left_behind") or []}
    rows = []
    for raw in out.get("worktrees") or []:
        row = dict(raw)
        hit = by_path.get(str(row.get("path")))
        if hit is not None:
            row["state"] = str(hit.get("state") or "restored")
            row.pop("reason", None)
            if hit.get("reason"):
                row["reason"] = str(hit["reason"])
            for key in ("dest_path", "branch", "mode", "recipes"):
                if hit.get(key):
                    row[key] = hit[key]
        elif str(row.get("path")) in left:
            row["state"] = "left_behind"
            row["reason"] = str(left[str(row.get("path"))].get("reason") or "not_transferred")
        rows.append(row)
    out["worktrees"] = rows
    out["transferred"] = True
    out["state"] = str(restored.get("state") or "restored")
    if restored.get("notes"):
        out["notes"] = list(restored["notes"])
    return out


def resume_pending(server: Any) -> list[str]:
    """Relay start: retry every ``workset-in/<sid>`` still ``pending_restore``.

    Only a session this device now holds is restored (its directory exists under
    ``sessions/``); anything else waits. Returns the session ids retried.
    """
    base = _network(server.root) / "workset-in"
    retried: list[str] = []
    try:
        children = sorted(base.iterdir())
    except OSError:
        return retried
    for child in children:
        pending = _read_json(child / "pending.json")
        if not pending or not child.is_dir() or child.is_symlink():
            continue
        restored = pending.get("restored") or {}
        plans = pending.get("plans") or []
        if all(
            str(p.get("n")) in restored
            and restored[str(p.get("n"))].get("state") != "pending_restore"
            for p in plans
        ):
            continue
        if not (Path(server.root) / "sessions" / child.name).is_dir():
            continue
        try:
            restore_arrived(server, child.name)
            retried.append(child.name)
        except Exception:  # noqa: BLE001 — housekeeping
            logger.debug("workset: resume of %s failed", child.name, exc_info=True)
    return retried


def sweep_outgoing(root: Path | str, *, max_age_s: float) -> list[str]:
    """Age sweep of ``workset-out/<sid>/`` no in-flight move still names (owner side).

    Only this module's own staging is removed (journal + bundles); the departure
    journal and the snapshot refs stay.
    """
    from local_operator.session.placement import handoff_in_flight

    base = _network(root) / "workset-out"
    swept: list[str] = []
    try:
        children = list(base.iterdir())
    except OSError:
        return swept
    cutoff = time.time() - max_age_s
    for child in children:
        try:
            if child.is_symlink() or not child.is_dir() or child.stat().st_mtime >= cutoff:
                continue
            if handoff_in_flight(Path(root), child.name):
                continue
        except Exception:  # noqa: BLE001
            continue
        _remove_tree(child)
        swept.append(child.name)
    return swept


__all__ = [
    "AUDIT_WORKSET_DEPARTED",
    "AUDIT_WORKSET_RESTORE",
    "AUDIT_WORKSET_SERVE",
    "WorksetTransferRefused",
    "arrival_receipt",
    "arrived_dir",
    "destination_transfer",
    "discard_incoming",
    "discard_outgoing",
    "home_of",
    "incoming_staging",
    "mark_departed",
    "outgoing_dir",
    "promote_incoming",
    "ready_digest_matches",
    "restore_arrived",
    "resume_pending",
    "serve_fetch",
    "serve_plan",
    "source_prepare",
    "sweep_outgoing",
    "workset_digest",
]
