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

#: Anchored by ``fullmatch`` at every use, never ``match`` with ``$`` (security review
#: #4245 round 6, S-4): ``$`` also matches before a trailing newline, so a digest
#: ``sha256:<hex>\n`` would pass the shape check and only the hash compare refused it.
_HEX64: Final = re.compile(r"[0-9a-f]{64}")
_DIGEST: Final = re.compile(r"sha256:([0-9a-f]{64})")
_SHA: Final = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")

#: Next contiguous offset served per (session, lease, to_device, digest). A request at
#: exactly that offset continues an audited stream; anything else (offset 0, a gap, a
#: relay restart that forgot the cursor) is a NEW serve and is audited first.
_SERVE_CURSOR: dict[tuple[str, str, str, str], int] = {}
_SERVE_LOCK = threading.Lock()
#: S8: the decline reasons the SOURCE may learn. Both are decided by this device's
#: policy from the offer alone (path rules, URL rules), not by what exists here;
#: every state-dependent reason (absent, conflict, unsafe component...) is sent as
#: ``destination_declined`` and kept precisely in this device's ``pending.json``.
_COARSE_DECLINE: Final = frozenset({"restore_path_refused", "url_refused", "too_many_worktrees"})
#: Rows of an offer this device considers (the owner's manifest never comes close;
#: the same bound as the probe's ``MAX_PROBE_ROWS``).
MAX_OFFER_ROWS: Final = 64

# ------------------------------------------------- destination: import authority
#
# SECURITY-LEAD RULING (#4245, 2026-11-18, binding): "require explicit destination
# authorization for importing the working set, separate from session mobility (a
# receiver-controlled capability or equivalent narrowly scoped policy). Path
# allowlists constrain where writes land; they do not grant permission to accept a
# peer's data and writes. Bind that policy check to the authenticated peer,
# destination, lease and approved manifest; absent grant should leave the
# session-only move available without file import."
#
# The mechanism is the member capability ``workset_import``, granted ON THIS DEVICE to
# the SOURCE device (``lop network member grant <network> <source-id> workset_import``,
# a ``SELF_DECIDED_SCOPES`` write, so the receiving device records it without an admin
# row). ``move`` never implies it, and neither does ``admin`` (``types.ROLE_CAPABILITIES``).
# It is read from THIS device's own member row for the AUTHENTICATED link peer — never
# from a frame — at the probe (so the user is told before consenting), at the prepare
# (so an honest owner never offers), before every clone/plan/fetch of the transfer, and
# again at the restore after ``done`` (so a revoke in between stops the writes).
#
# THE RECALL IS ITS OWN AUTHORIZATION. ``--to local`` is initiated on THIS device and
# its manifest is confirmed by the operator sitting at it: the person whose home the
# files land in read the list and said yes, which is exactly the receiver-side
# decision the ruling asks for, scoped to that one move. A queued recall resumes
# through the owner's invite, which this device cannot tell from a push, so it falls
# back to the grant (fail-closed: without one the session moves and the files stay).

#: The capability name (``types.CAPABILITIES``).
IMPORT_CAPABILITY: Final = "workset_import"
#: Why this device accepted a peer's working files, recorded in ``pending.json``.
IMPORT_BY_GRANT: Final = "grant"
IMPORT_BY_RECALL: Final = "recall"


def import_granted(root: Path | str, device_id: str, *, network_id: str = "") -> bool:
    """Whether THIS device's own record lets ``device_id`` send it working files.

    Reads every network record this device holds (or just ``network_id``) for an
    ACTIVE member row of ``device_id`` that carries ``workset_import``. Never raises:
    an unreadable record grants nothing.
    """
    if not device_id:
        return False
    from local_operator.network import store

    try:
        records = store.list_networks(Path(root))
    except Exception:  # noqa: BLE001 — unreadable authority is no authority
        return False
    for record in records:
        if network_id and str(record.network_id) != network_id:
            continue
        member = record.member(device_id)
        if member is not None and member.active and IMPORT_CAPABILITY in member.capabilities:
            return True
    return False


def import_authority(
    root: Path | str, device_id: str, *, recall: bool, network_id: str = ""
) -> str:
    """``recall`` / ``grant`` when this device accepts ``device_id``'s files, else ``''``."""
    if recall:
        return IMPORT_BY_RECALL
    return IMPORT_BY_GRANT if import_granted(root, device_id, network_id=network_id) else ""


def import_remedy(network_id: str, device_id: str) -> str:
    """The exact command, run ON THE DESTINATION, that lets ``device_id`` send files.

    Ids only (``member grant`` resolves a network id and a device id; a name can be
    ambiguous or renamed), each shell-quoted because both arrive from a peer's frame.
    """
    import shlex

    return " ".join(
        [
            "lop network member grant",
            shlex.quote(network_id or "<network>"),
            shlex.quote(device_id or "<device>"),
            IMPORT_CAPABILITY,
        ]
    )


#: ``staging/<sid>.workset/cloned.json``: the primaries THIS move cloned (S1c).
CLONED_JOURNAL: Final = "cloned.json"
#: Security review #4245 S4: the ``prefix`` resume check is a byte-granular content
#: oracle unless bounded. A prefix must be a whole number of these (the destination
#: truncates its ``.part`` down to one before asking), it is audited like a span, and
#: each (session, lease, device, artifact) gets at most ``_MAX_PREFIX_QUERIES``. The
#: alignment is the span size itself, so a resume loses at most one span. A module
#: value so a test can align a small artifact; never set from a frame.
_RESUME_ALIGN: int = 512 * 1024
_MAX_PREFIX_QUERIES: Final = 2
_PREFIX_QUERIES: dict[tuple[str, str, str, str], int] = {}
#: S5: plans one prepared move answers — the first, plus the single ``have=[]`` re-plan
#: W5 needs. Each plan stages bundles on THIS device's disk; more would let the
#: destination fill it with permutations of ``have``.
_MAX_PLANS: Final = 2


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


#: Confirmed recall scopes kept per session (each re-run of a preflight adds one).
_MAX_CONFIRMABLE_SCOPES: Final = 8


def _confirmable_path(root: Path | str, session_id: str) -> Path:
    """Destination: ``<config>/network/workset-consent/<sid>.json``."""
    return _network(root) / "workset-consent" / f"{_sid(session_id)}.json"


def record_confirmable(
    root: Path | str, session_id: str, block: Mapping[str, Any], *, self_device: str
) -> bool:
    """Remember WHICH rows a recall preflight showed this device's operator (S13).

    WHY (#4245 security round 3, S13): on a recall the authority to write into this
    home is "the person whose home the files land in read the list and said yes" —
    but the binding the offer carries is computed by the OWNER, which knows the
    confirmed digest and the lease, so it can bind any rows it likes. The only
    record of what the operator actually read is the preflight answer this device
    received, so it is kept here, keyed by the scope digest the operator confirms,
    and ``destination_transfer`` refuses any offered row outside it. Planned rows
    only: nothing else is ever offered. Bounded to the newest few digests.

    THE DIGEST IS RECOMPUTED, NOT BELIEVED (#4245 round 4, S13/M1). The block's
    ``scope_digest`` is the owner's own label, so a record keyed by it alone let a
    hostile owner answer a later preflight with the confirmed label and more rows,
    and the transfer read those back as "confirmed". The digest is now rebuilt here
    from the block's rows with ``to_device`` = ``self_device`` (this device's mesh
    id: a digest minted for another device is refused too), and a block whose
    label does not match is not recorded. A digest then names exactly one row set,
    so the record is WRITE-ONCE per digest: an equal re-record is a no-op, and a
    different row set under a recorded digest (only a tampered file or a collision
    can produce one) keeps the first and is logged. Returns whether ``block``'s rows
    are now the record for its digest.
    """
    from local_operator.network import workset

    digest = str(block.get("scope_digest") or "")
    if not digest or not self_device:
        return False
    if workset.block_scope_digest(block, to_device=self_device) != digest:
        logger.warning(
            "workset: %s's preflight names digest %s but its rows hash otherwise for "
            "this device; not recorded, so a recall confirming it carries no files",
            session_id,
            digest[:19],
        )
        return False
    rows = sorted(
        {
            (str(r.get("path") or ""), str(r.get("repo") or ""))
            for r in block.get("worktrees") or []
            if isinstance(r, Mapping) and r.get("state") == "planned"
        }
    )
    path = _confirmable_path(root, session_id)
    _private_dir(path.parent)
    scopes = dict((_read_json(path) or {}).get("scopes") or {})
    wanted = [list(row) for row in rows]
    if digest in scopes:
        if scopes[digest] != wanted:
            logger.warning(
                "workset: %s already has a different row set recorded for %s; kept the first",
                session_id,
                digest[:19],
            )
            return False
        return True
    scopes[digest] = wanted
    while len(scopes) > _MAX_CONFIRMABLE_SCOPES:
        scopes.pop(next(iter(scopes)))
    _write_json(path, {"version": JOURNAL_VERSION, "scopes": scopes})
    return True


def confirmed_rows(
    root: Path | str, session_id: str, digest: str
) -> frozenset[tuple[str, str]] | None:
    """The ``(path, repo)`` rows the preflight for ``digest`` showed, or ``None``."""
    scopes = (_read_json(_confirmable_path(root, session_id)) or {}).get("scopes")
    if not digest or not isinstance(scopes, Mapping) or digest not in scopes:
        return None
    return frozenset(
        (str(r[0]), str(r[1])) for r in scopes[digest] or [] if isinstance(r, list) and len(r) == 2
    )


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
        "cwd": (engine.home_relative(cwd, home=home) or "") if cwd else "",
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
    plans_so_far = int(journal.get("plans") or 0)
    if plans_so_far >= _MAX_PLANS:
        return _refused(
            "plan_limit",
            "this move already planned its working files and re-planned them once",
            session_id,
        )
    journal["plans"] = plans_so_far + 1
    _write_json(outgoing_dir(server.root, session_id) / "journal.json", journal)
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
        _prune_artifacts(out, keep=())
        return _refused(
            "working_set_over_cap",
            f"the working files are {total} bytes once bundled, over this device's "
            f"cap of {cap} (mesh.move.working_set_max_bytes), so nothing was moved",
            session_id,
        )
    journal["artifacts"] = artifacts
    journal["plan_left_behind"] = left
    _write_json(out / "journal.json", journal)
    # A re-plan supersedes the earlier artifact of the worktree it re-bundled.
    _prune_artifacts(
        out, keep=[str(a["digest"]).removeprefix("sha256:") for a in artifacts.values()]
    )
    return {
        "result": "planned",
        "session_id": session_id,
        "artifacts": reply_artifacts,
        "left_behind": left,
    }


def _prune_artifacts(out: Path, *, keep: Sequence[str]) -> None:
    """S5: unlink staged artifacts (and their sidecars) the current plan does not name.

    Only ``<64 hex>`` and ``<64 hex>.json`` regular files directly in this move's own
    ``workset-out/<sid>/`` — never the journal, never a symlink's target.
    """
    wanted = set(keep)
    try:
        children = list(out.iterdir())
    except OSError:
        return
    for child in children:
        stem = child.name.removesuffix(".json")
        if not _HEX64.fullmatch(stem) or stem in wanted or child.is_symlink():
            continue
        try:
            child.unlink()
        except OSError:
            logger.debug("workset: could not prune %s", child, exc_info=True)


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
    kind: str = "span",
) -> bool:
    """Write ONE durable ``session.workset.serve`` row; ``True`` only if it is on disk.

    The caller serves ZERO bytes on ``False`` (design §8.3): an egress an incident
    cannot reconstruct is the loss the audit trail exists to prevent, so any doubt is a
    refusal.

    WHY ``AuditLog.record_required`` AND NOT ``record``: ``record`` never raises and
    only flushes on a timer, so it cannot gate bytes. ``record_required`` (PR #4228)
    is the log's own fail-closed admission: it refuses a disabled or already-degraded
    log, holds the writer's lock across the append and its acknowledgement so a
    concurrent row cannot stand in for this one, and answers ``True`` only once THIS
    row is published and fsynced. This function used to reimplement that check from
    the writer's privates (``_write_lock``, ``syncs``, ``publication_of``) while #4228
    was unmerged; the swap removes that second copy of the durability contract and
    adds the degraded-log refusal the stand-in lacked. A contiguous continuation does
    not come here (one row covers the stream); ``serve_fetch`` re-checks the same
    admission predicate, ``AuditLog.admits_required``, before each one.

    The event is in ``AuditLog.DURABLE_EVENTS`` (``record_required`` refuses anything
    else) and every key below is in its ``DETAIL_KEYS`` allowlist, so the row is
    written as sent. Metadata is bounded: no payload, no secret, no file content —
    ids, sizes and one home-relative path, cut to the log's per-value cap here so the
    row holds what this function sent rather than a silently shortened copy.
    """
    from local_operator.network.audit import MAX_DETAIL_VALUE_CHARS, AuditEvent

    log = getattr(server, "audit", None)
    if log is None:
        return False
    try:
        return log.record_required(
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
                    "path": path[:MAX_DETAIL_VALUE_CHARS],
                    "kind": kind,
                },
            )
        )
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

    AUDIT: the first span of a stream (offset 0, or any offset that does not continue
    this process's cursor — a resume) writes ONE durable ``session.workset.serve`` row
    before its first byte, and that row covers every contiguous continuation after it.
    Each continuation still re-checks the log's health (``AuditLog.admits_required``,
    #4245 security review S-1): a log disabled or degraded mid-stream refuses the next
    span with ``audit_unavailable`` and zero bytes, and writes no row — the existing
    row already covers the bytes sent before the hole.
    """
    from local_operator.network import sync as sync_mod

    session_id = str(frame.get("session_id") or "")
    journal, refusal = _authorised_journal(server, link, frame, session_id)
    if refusal is not None or journal is None:
        return refusal or _refused("refused", "no working set", session_id)
    match = _DIGEST.fullmatch(str(frame.get("digest") or ""))
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
            epoch = str(journal.get("lease_epoch") or "")
            key = (session_id, epoch, str(link.device_id), digest)
            row_path = next(
                (
                    str(r.get("path") or "")
                    for r in journal.get("worktrees") or []
                    if str(r.get("n")) == staged[digest]
                ),
                "",
            )
            if prefix_len is not None:
                if prefix_len > size:
                    return _refused("bad_request", "prefix is longer than the artifact", session_id)
                # S4: only a span-aligned prefix, a bounded number of times, and the
                # answer is a SERVE — audited before it is computed, like any span.
                if prefix_len == 0 or prefix_len % _RESUME_ALIGN:
                    return _refused(
                        "bad_request", "a resume prefix must be a whole number of spans", session_id
                    )
                with _SERVE_LOCK:
                    asked = _PREFIX_QUERIES.get(key, 0)
                    if asked >= _MAX_PREFIX_QUERIES:
                        return _refused(
                            "bad_request", "this artifact was already resumed enough", session_id
                        )
                    _PREFIX_QUERIES[key] = asked + 1
                if not _audit_workset_serve(
                    server,
                    session_id=session_id,
                    to_device=str(link.device_id),
                    lease_epoch=epoch,
                    digest=digest,
                    size=size,
                    offset=prefix_len,
                    path=row_path,
                    kind="prefix",
                ):
                    return _refused(
                        "audit_unavailable",
                        "this device could not record the transfer in its audit log, so "
                        "no working file was sent",
                        session_id,
                    )
                with _SERVE_LOCK:
                    # The audited prefix row covers the resume that follows it.
                    _SERVE_CURSOR[key] = prefix_len
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
            with _SERVE_LOCK:
                continuing = offset > 0 and _SERVE_CURSOR.get(key) == offset
            if continuing:
                # No new row (the stream's first row covers this span), but no bytes
                # into a trail that has since become disabled or holed (S-1).
                log = getattr(server, "audit", None)
                if log is None or not log.admits_required():
                    return _refused(
                        "audit_unavailable",
                        "this device could not record the transfer in its audit log, so "
                        "no working file was sent",
                        session_id,
                    )
            else:
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
        # S4: the owner answers only a span-aligned prefix, so drop the partial span.
        have = min(part.stat().st_size, expected_bytes)
        have -= have % _RESUME_ALIGN
        if have > 0:
            pfd = os.open(part, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
            try:
                os.ftruncate(pfd, have)
            finally:
                os.close(pfd)
            local = _sha256(part)
            try:
                answer = _ask(transport, {**base, "prefix": have})
            except WorksetTransferRefused:
                answer = {}  # an owner that will not confirm: start over at 0
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
    authority: str = "",
    recheck: Any = None,
    approved_scope: str = "",
    self_device: str = "",
    network_id: str = "",
    confirmed: frozenset[tuple[str, str]] | None = None,
) -> str:
    """W1–W5 on the destination. Returns the ``workset_digest`` for ``ready``.

    IMPORT AUTHORITY FIRST (security-lead ruling, see ``IMPORT_CAPABILITY``).
    ``authority`` is the caller's reading for the AUTHENTICATED owner: ``recall`` (this
    device asked and its operator confirmed) or ``grant`` (``workset_import`` on this
    device's row for it); anything else refuses before a single clone, plan or fetch.
    A ``grant`` is re-read through ``recheck()`` (default: :func:`import_granted`)
    before W1, after each W1 row, before each plan and before each fetch, so a revoke
    mid-transfer stops it. ``approved_scope`` + ``self_device``: the confirmed scope
    digest and this device's id; the offer's ``binding`` must equal
    ``workset.transfer_binding`` over them, this lease and the offered snapshot ids, so
    the decision is bound to the approved manifest for THIS destination and lease. It
    is journaled with them in ``pending.json`` for the restore's re-check
    (:func:`restore_arrived`).

    Raises :class:`WorksetTransferRefused` (the caller rolls the move back exactly as a
    failed session copy: nothing committed, ``changed: False``). Writes
    ``staging/<sid>.workset/pending.json`` with one restore plan per artifact, for the
    restore that runs after ``done``.

    THE DESTINATION DECIDES (security review #4245). Every path and URL below comes
    from the source's offer, and a source holding ``move`` may be hostile, so before
    anything is cloned or fetched this device applies its OWN policy:

    * S1 — a repository lands only where ``engine.primary_path_refusal`` allows
      (plain components, no config/autostart roots, under
      ``mesh.move.working_set_roots``); a worktree only under ``<primary>/.worktrees``
      (``engine.restore``). Refused → ``left_behind`` ``restore_path_refused``.
    * S2 — a clone URL must pass ``engine.clone_url_refusal``; the https form of an
      ssh remote is derived HERE (an offered rewrite is used only if it equals that).
    * S1c — a clone this move makes is journaled in ``cloned.json`` as soon as it
      exists, so the rollback (``discard_incoming``) removes it, and only it.
    * S8 — what goes back to the source is coarse (``_COARSE_DECLINE``); the precise
      reason stays in this device's ``pending.json``. S12 (round 2), accepted residual:
      whether a home path exists is still observable (an absent path may be cloned, an
      existing foreign one is declined), but only to a peer this device granted
      ``workset_import`` — the receiver-side consent the review named as the fix —
      and the default clone policy limits clones to folders already holding
      repositories, so the observable set is those folders' direct children.
    * S6/R3/R4 — the cap and free space are checked on the first plan AND after a
      re-plan, and the checkout's measured size is checked against the home
      filesystem before ``ready``.
    """
    from local_operator.network import workset, workset_probe
    from local_operator.network import workset_transfer as engine

    def authorised() -> None:
        if authority == IMPORT_BY_RECALL:
            return
        ok = authority == IMPORT_BY_GRANT and bool(
            recheck() if recheck is not None else import_granted(server.root, owner_device)
        )
        if not ok:
            raise WorksetTransferRefused(
                "working_set_import_not_granted",
                "this device has not allowed that device to send it working files "
                f"(run `{import_remedy(network_id, owner_device)}` here to allow it)",
            )

    authorised()
    all_rows = [r for r in offer.get("worktrees") or [] if isinstance(r, Mapping)]
    offered_refusal = WorksetTransferRefused(
        "working_set_transfer_failed",
        "the files offered are not the ones confirmed for this move",
    )
    # S16 (#4245 security round 3): the production caller always names this device,
    # and then the binding is MANDATORY — an owner that omits the scope or the
    # binding no longer skips the check below (it was ``if approved_scope``).
    if self_device and all_rows and not (approved_scope and offer.get("binding")):
        raise offered_refusal
    # S13: on a recall the binding is computed by the party it binds, so it cannot
    # prove what this device's operator read. ``confirmed`` (``confirmed_rows``) is
    # the preflight this device itself received for the confirmed digest; every
    # offered row must be one of its planned rows, and with no such record a recall
    # carries no files. On a ``grant`` the binding stays what it is: anti-replay and
    # anti-misdirection for an honest owner the operator already chose to trust.
    if authority == IMPORT_BY_RECALL and self_device and all_rows:
        if confirmed is None or any(
            (str(r.get("path") or ""), str(r.get("repo") or "")) not in confirmed for r in all_rows
        ):
            raise offered_refusal
    if approved_scope and self_device:
        expected = workset.transfer_binding(
            approved_scope,
            lease_epoch=lease_epoch,
            to_device=self_device,
            snapshots=[
                {k: r.get(k) for k in ("path", "head", "index", "worktree")} for r in all_rows
            ],
        )
        if str(offer.get("binding") or "") != expected:
            raise offered_refusal
    home = home_of(server)
    into = _private_dir(incoming_staging(server.root, session_id))
    roots = workset.restore_roots(Path(server.root))
    forbid = [Path(server.root)]
    rows = all_rows[:MAX_OFFER_ROWS]
    rewrite_notes: list[str] = []
    have: dict[str, list[str]] = {}
    decline: dict[str, str] = {}
    local_reason: dict[str, str] = {}
    # Review N2: rows past the bound are DECLINED by name, so the owner never bundles
    # them and the ready digests still agree; a silent cut rolled the whole move back
    # with a generic failure.
    for row in all_rows[MAX_OFFER_ROWS:]:
        decline[str(row.get("n"))] = "too_many_worktrees"
        local_reason[str(row.get("n"))] = "too_many_worktrees"
    by_n_all = {str(r.get("n")): r for r in all_rows}
    primaries: dict[str, dict[str, Any]] = {}
    # S9 (round 2): with ``mesh.move.working_set_roots`` unset, an absent repository is
    # cloned only beside repositories the user already keeps (``clone_parent_refusal``).
    explicit_roots = workset.restore_roots_configured(Path(server.root))
    for row in rows:
        n = str(row.get("n"))
        remote_url = str(row.get("remote_url") or "")
        repo_rel = str(row.get("repo") or "")
        use_url = remote_url
        res = None
        cloned = False
        if engine.primary_path_refusal(home, repo_rel, roots=roots, forbid=forbid):
            res = engine.PrimaryResolution("unsafe", None, reason="restore_path_refused")
        elif remote_url:
            res = engine.resolve_primary(use_url, repo_rel, home=home)
        if res is not None and res.state == "absent":
            # THE HTTPS REWRITE (design §4 W1): an ssh remote this device has no key for
            # is cloned over its https form instead, with THIS device's own
            # credentials. Tried in order — the original URL, then the rewrite — so a
            # destination that can use ssh keeps it. The rewrite is DERIVED HERE from
            # ``remote_url`` (``workset_probe.https_rewrite``: same host, ssh/scp forms
            # only); the offer's ``https_rewrite.url`` is ignored, so the source cannot
            # name a second URL of its choosing (S2). Repo-local by construction (the
            # fresh clone's origin) and named in the receipt's notes.
            alt = workset_probe.https_rewrite(remote_url)
            candidates = [remote_url] + ([alt] if alt and alt != remote_url else [])
            allowed = [c for c in candidates if not engine.clone_url_refusal(c)]
            res = engine.PrimaryResolution(
                "absent", None, reason="destination_cannot_clone" if allowed else "url_refused"
            )
            if allowed and not explicit_roots and engine.clone_parent_refusal(home, repo_rel):
                # State-dependent (what the parent holds), so the owner hears only
                # ``destination_declined`` (S8); the precise reason stays here.
                res = engine.PrimaryResolution("absent", None, reason="clone_parent_refused")
                allowed = []
            for candidate in allowed:
                try:
                    got = engine.clone_primary(
                        candidate,
                        repo_rel,
                        home=home,
                        pack_bytes=int(row.get("pack_bytes") or 0),
                        session_id=session_id,
                        lease_epoch=lease_epoch,
                        make_parents=explicit_roots,
                    )
                except engine.MoveWorksetError as exc:
                    if exc.reason == "insufficient_space":
                        raise WorksetTransferRefused(
                            "insufficient_space",
                            f"this device does not have room to clone {repo_rel} "
                            f"({exc.detail}), so nothing was moved",
                        ) from exc
                    continue
                if got.cloned:
                    _journal_clone(into, repo_rel, session_id, lease_epoch)
                if got.state == "present":
                    res, use_url, cloned = got, candidate, True
                    if candidate != remote_url:
                        rewrite_notes.append(f"{repo_rel}: {remote_url} -> {candidate}")
                    break
        primaries[n] = {"res": res, "url": use_url, "cloned": cloned}
        authorised()
        if res is not None and res.state == "present" and res.path is not None:
            tips = [str(t) for t in row.get("tips") or [] if _SHA.match(str(t))][:64]
            have[n] = engine.compute_have(res.path, tips, remote=res.remote, session_id=session_id)
        else:
            reason = (res.reason if res is not None else "") or "destination_cannot_clone"
            local_reason[n] = reason
            decline[n] = reason if reason in _COARSE_DECLINE else "destination_declined"
    authorised()
    planned = _plan(transport, session_id, lease_epoch, have, decline)
    artifacts = {
        str(a.get("n")): a for a in planned.get("artifacts") or [] if isinstance(a, Mapping)
    }
    left = [dict(r) for r in planned.get("left_behind") or [] if isinstance(r, Mapping)]
    _check_room(server, into, artifacts)
    plans: list[dict[str, Any]] = []
    mapping: list[tuple[str, list[str]]] = []
    checkout_total = 0
    replanned = False
    for row in rows:
        n = str(row.get("n"))
        art = artifacts.get(n)
        if art is None:
            continue
        digest = str(art.get("digest") or "")
        if not _DIGEST.fullmatch(digest):
            raise WorksetTransferRefused(
                "working_set_transfer_failed", "the owner named a malformed artifact"
            )
        info = primaries.get(n) or {}
        res = info.get("res")
        authorised()
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
                # W5: re-plan ONCE (for the whole move, which is all the owner allows,
                # S5) with ``have=[]`` for this worktree; a second failure rolls the
                # whole move back (nothing committed anywhere).
                if replanned:
                    raise WorksetTransferRefused(
                        "working_set_transfer_failed",
                        f"the bundle for {row.get('path')} did not verify here",
                    )
                replanned = True
                authorised()
                again_plan = _plan(transport, session_id, lease_epoch, {**have, n: []}, decline)
                replanned_artifacts = {
                    str(a.get("n")): a
                    for a in again_plan.get("artifacts") or []
                    if isinstance(a, Mapping)
                }
                merged = {**artifacts, **replanned_artifacts}
                again = replanned_artifacts.get(n)
                if not again:
                    raise WorksetTransferRefused(
                        "working_set_transfer_failed", "the re-plan carried nothing"
                    )
                digest = str(again.get("digest") or "")
                if not _DIGEST.fullmatch(digest):
                    raise WorksetTransferRefused(
                        "working_set_transfer_failed", "the owner named a malformed artifact"
                    )
                # R3: the full bundle is bigger; this device's cap and free space are
                # checked again before a byte of it is fetched.
                _check_room(server, into, merged)
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
                artifacts = merged
            # S6/R4: what the checkout will WRITE, measured from the bundle's own
            # objects (not a number the source sent).
            worktree_sha = str(row.get("worktree") or "")
            if _SHA.match(worktree_sha):
                try:
                    checkout_total += engine.measure_checkout(
                        res.path,
                        fetched,
                        engine.snapshot_ref(session_id, int(row.get("n") or 0), "worktree"),
                        into,
                    )
                except engine.MoveWorksetError as exc:
                    raise WorksetTransferRefused(
                        "working_set_transfer_failed",
                        f"the bundle for {row.get('path')} could not be read here",
                    ) from exc
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
    free_home = shutil.disk_usage(home).free
    if checkout_total + engine.CHECKOUT_RESERVE_BYTES > free_home:
        raise WorksetTransferRefused(
            "insufficient_space",
            f"this device has {free_home} bytes free where the files are restored and "
            f"they need about {checkout_total + engine.CHECKOUT_RESERVE_BYTES}, so nothing "
            "was moved",
        )
    by_n = by_n_all
    # WHAT THE SOURCE'S RECEIPT MAY LEARN ABOUT EACH ROW THAT DID NOT TRAVEL (#4245
    # agent review round 3, B2). Captured BEFORE the S8 override below, so a row this
    # device declined carries the exact coarse code the owner was already sent
    # (``decline``) and a row the owner's own plan left behind keeps the owner's own
    # reason — never one of this device's precise, state-dependent reasons.
    # ``destination_outcome`` reads it: without it every offered row reached the
    # source's receipt as "Sent", including ones that never left the source.
    source_left: dict[str, str] = {}
    for item in left:
        n = str(item.get("n"))
        path = str(item.get("path") or by_n.get(n, {}).get("path") or "")
        if path:
            source_left[path] = (
                decline.get(n) or str(item.get("reason") or "") or "destination_declined"
            )
    for n, coarse in decline.items():
        path = str(by_n.get(n, {}).get("path") or "")
        if path:
            source_left.setdefault(path, coarse)
    # S8: the owner was told only a coarse reason; this device keeps the precise one.
    # Every declined row is recorded here even if the owner's reply omitted it.
    for item in left:
        n = str(item.get("n"))
        if n in local_reason:
            item["reason"] = local_reason[n]
            item.setdefault("path", by_n.get(n, {}).get("path"))
    listed = {str(item.get("n")) for item in left}
    for n, reason in local_reason.items():
        if n not in listed:
            left.append(
                {"n": int(n) if n.isdigit() else n, "path": by_n[n].get("path"), "reason": reason}
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
            "source_left": source_left,
            "notes": rewrite_notes,
            "cwd": str(offer.get("cwd") or ""),
            "restored": {},
            # What authorised these writes, bound to this owner, this device, this
            # lease and the offer's binding; ``restore_arrived`` re-checks it.
            "import": {
                "authority": authority,
                "from_device": owner_device,
                "to_device": self_device,
                "network_id": network_id,
                "lease_epoch": lease_epoch,
                "binding": str(offer.get("binding") or ""),
            },
        },
    )
    return digest_all


def _check_room(server: Any, into: Path, artifacts: Mapping[str, Mapping[str, Any]]) -> None:
    """The destination's own cap and staging free space for ``artifacts`` (W1/W5, R3)."""
    from local_operator.network import workset

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


def _journal_clone(into: Path, repo_rel: str, session_id: str, lease_epoch: str) -> None:
    """S1c: record a clone THIS move made, the moment it exists, for the rollback."""
    path = into / CLONED_JOURNAL
    record = _read_json(path) or {}
    clones = [c for c in record.get("clones") or [] if isinstance(c, dict)]
    clones.append({"path": repo_rel, "session_id": session_id, "lease_epoch": lease_epoch})
    _write_json(path, {"clones": clones})


def discard_incoming(root: Path | str, session_id: str, *, home: Path | None = None) -> None:
    """The destination rollback's half: remove ``staging/<sid>.workset/``.

    With ``home``, first remove every primary THIS move cloned (``cloned.json``; S1c):
    a refused or rolled-back move must not leave a fresh repository behind it. Each one
    is removed only if its own marker still names this session and lease
    (``engine.remove_own_clone``); anything else stays.
    """
    if home is not None:
        from local_operator.network import workset
        from local_operator.network import workset_transfer as engine

        try:
            record = _read_json(incoming_staging(root, session_id) / CLONED_JOURNAL) or {}
        except ValueError:
            record = {}
        # S-3 (round 6): the same destination policy a clone target passed, re-read
        # here so ``remove_own_clone`` vets every journal row itself.
        roots = workset.restore_roots(Path(root))
        for clone in record.get("clones") or []:
            if not isinstance(clone, dict):
                continue
            try:
                engine.remove_own_clone(
                    home,
                    str(clone.get("path") or ""),
                    session_id=str(clone.get("session_id") or ""),
                    lease_epoch=str(clone.get("lease_epoch") or ""),
                    config_root=Path(root),
                    roots=roots,
                )
            except OSError:
                logger.warning("workset: could not remove the clone this move made", exc_info=True)
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
    from local_operator.network import workset

    home = home_of(server)
    roots = workset.restore_roots(Path(server.root))
    restored: dict[str, Any] = dict(pending.get("restored") or {})
    for raw in pending.get("plans") or []:
        key = str(raw.get("n"))
        if key in restored and restored[key].get("state") != "pending_restore":
            continue
        plan = engine.RestorePlan(**raw)
        # The authority is re-read at EACH row's boundary, immediately before this row
        # touches the tree, not once for the whole set: one restore can run for a while
        # (several worktrees, each a bundle fetch-and-checkout), and a `workset_import`
        # revoke landing mid-way must stop every row that has not started yet. Rows
        # already restored stay restored (their writes are done and recorded); the
        # rest stay staged as ``pending_restore`` / ``import_not_granted`` so a
        # re-grant and the next resume finish them from the same bundles.
        if not _import_still_authorised(server, pending):
            # A revoke between the fetch and the restore stops the writes; the bundle
            # stays staged, so a re-grant and the next resume restore it.
            restored[key] = asdict(
                engine.RestoreReport("pending_restore", plan.path, reason="import_not_granted")
            )
            pending["restored"] = restored
            _write_json(directory / "pending.json", pending)
            continue
        try:
            report = engine.restore(
                directory, plan, home=home, config=Path(server.root), roots=roots
            )
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
            if _HEX64.fullmatch(child.name) or child.name.endswith(".part"):
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


def _import_still_authorised(server: Any, pending: Mapping[str, Any]) -> bool:
    """The restore's re-check of what authorised the transfer (``pending["import"]``).

    The record must name the same owner and lease as the plan it rides with; a
    ``grant`` must still be on this device's row for that owner (a revoke after the
    fetch stops the writes), a ``recall`` stands (this device's operator confirmed it).
    A pending journal without the record authorises nothing.
    """
    stamp = pending.get("import")
    if not isinstance(stamp, Mapping):
        return False
    if str(stamp.get("from_device") or "") != str(pending.get("from_device") or ""):
        return False
    if str(stamp.get("lease_epoch") or "") != str(pending.get("lease_epoch") or ""):
        return False
    authority = str(stamp.get("authority") or "")
    if authority == IMPORT_BY_RECALL:
        return True
    if authority != IMPORT_BY_GRANT:
        return False
    return import_granted(
        server.root,
        str(stamp.get("from_device") or ""),
        network_id=str(stamp.get("network_id") or ""),
    )


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
    # Review R1: the DEEPEST restored tree containing the cwd wins. Rows come in plan
    # order (sorted by path), so a primary ``~/proj`` precedes its own
    # ``~/proj/.worktrees/feat``; taking the first match sent a session working in
    # ``feat`` to the primary's restored copy instead.
    matches = [
        row
        for row in rows
        if row.get("state") == "restored"
        and str(row.get("path") or "").startswith("~/")
        and row.get("dest_path")
        and (
            cwd_rel == str(row.get("path"))
            or cwd_rel.startswith(str(row.get("path")).rstrip("/") + "/")
        )
    ]
    if not matches:
        return
    row = max(matches, key=lambda r: len(str(r.get("path"))))
    src, dest = str(row.get("path") or ""), str(row.get("dest_path") or "")
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
    # THE DURABLE RECORD FOLLOWS THE RESTORE TOO (review #4403 r2, R2-n1). The
    # promote's remap ran in staging, before this tree existed, so it left
    # ``cwd.json`` blank; only the marker above named the restored tree, and a session
    # with no marker (exec/TUI) would read "cwd unknown" on its next move. Same rule
    # as every other durable write: contained under this home (``carry.contained_cwd``)
    # and written through the one safe writer, or not at all.
    try:
        from local_operator.session.creation import write_session_cwd

        contained = carry.contained_cwd(str(target), home=str(home), config_root=server.root)
        if contained:
            write_session_cwd(
                Path(server.root) / "sessions" / session_id, contained, source="runtime"
            )
    except Exception:  # noqa: BLE001 — advice for the next move, never a failure
        logger.debug("workset: could not record %s's restored cwd", session_id, exc_info=True)


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
            for key in ("dest_path", "branch", "mode", "recipes", "branch_moved", "remapped"):
                if hit.get(key):
                    row[key] = hit[key]
            # QA Q1: a row that landed elsewhere (beside an occupied path, under the
            # primary's ``.worktrees``) says so, whichever restore path produced it.
            dest = str(row.get("dest_path") or "")
            if dest and dest != str(row.get("path") or "") and not row.get("remapped"):
                row["remapped"] = [str(row.get("path") or ""), dest]
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


#: The row states a destination may report back to the source on ``done`` (offload
#: receipt fix, #4245). A closed set: the source renders each through its own table.
OUTCOME_ROW_STATES = frozenset({"transferred", "restored", "pending_restore", "left_behind"})


def destination_left(root: Path | str, session_id: str) -> dict[str, str]:
    """The rows of ``session_id``'s arrived working set that did NOT travel here.

    ``{source path: coarse reason}`` exactly as ``destination_transfer`` journaled it
    for the source (``source_left``): this device's declines as the coarse code the
    owner was already told, the owner's own plan omissions with the owner's reason.
    Empty when nothing arrived or the journal predates the field.
    """
    pending = _read_json(arrived_dir(root, session_id) / "pending.json")
    raw = (pending or {}).get("source_left")
    if not isinstance(raw, Mapping):
        return {}
    return {str(path): str(reason) for path, reason in raw.items() if path and reason}


def destination_outcome(
    block: Mapping[str, Any] | None, left: Mapping[str, str] | None = None
) -> dict[str, Any] | None:
    """What THIS destination tells the source about the working set, on ``done``.

    WHY IT EXISTS: an offload's receipt is built on the SOURCE, which before this
    showed its own pre-invite block (``transferred: false``) — with the grant the files
    landed here while the source's receipt said they stayed. The ``done`` frame is the
    one authenticated channel back (it already carries the scheduled-state block).
    The restore runs AFTER ``done`` (design §4 phase 4, so a slow restore never holds
    the source's completion wait), so a planned row reads ``transferred``: its
    artifacts arrived and verified against the digest the source committed on. Only
    ``path`` / ``state`` / ``reason`` per row travel — never a recipe, a destination
    path or any text the source would display as a command.

    ``left`` (:func:`destination_left`) is what this device's own transfer did NOT
    take: a planned row named there reads ``left_behind`` with that coarse code
    instead of ``transferred`` (review round 3, B2: a declined row was reported to the
    source as sent while its files never left the source).
    """
    if not isinstance(block, Mapping):
        return None
    declined = left or {}
    rows: list[dict[str, Any]] = []
    for raw in block.get("worktrees") or []:
        if not isinstance(raw, Mapping):
            continue
        path = str(raw.get("path") or "")
        state = str(raw.get("state") or "")
        reason = str(raw.get("reason") or "")
        if state in ("planned", "transferring"):
            if path in declined:
                state, reason = "left_behind", declined[path]
            else:
                state = "transferred"
        if state not in OUTCOME_ROW_STATES:
            continue
        row: dict[str, Any] = {"path": path, "state": state}
        if reason:
            row["reason"] = reason
        rows.append(row)
    return {"transferred": True, "worktrees": rows}


def source_receipt(
    planned: Mapping[str, Any], outcome: Mapping[str, Any] | None, *, sent_digest: str = ""
) -> dict[str, Any]:
    """The SOURCE's receipt block for an offload that carried its working set.

    ``planned`` is this device's own pre-invite block (computed before the prepare
    decided the files travel, so its rows read as staying); ``outcome`` is the
    destination's ``done``-frame answer or ``None`` when none arrived.

    THE PEER SUPPLIES NO TEXT (same threat model as
    ``workset_probe.scrub_peer_block``): rows are matched by this device's OWN paths,
    a state outside :data:`OUTCOME_ROW_STATES` is ignored, a reason survives only as
    a code ``workset_text`` renders through its closed table (an unknown code shows as
    the code, never a command), and nothing else from the peer is kept — so a hostile
    destination can change which of our own rows reads as arrived, never what the
    receipt tells the user to run or where to look. ``side: "source"`` selects the
    source-perspective sentences. No answer reads ``outcome_unknown`` ("check on
    <dest>"), never "stayed": the source committed, so the files may well be there.

    THE PLAN THAT WAS SENT, NOT THE PRE-INVITE ONE (QA Q-2, #4380): ``planned`` says
    ``carries_files: false`` with the "files stay" digest, which a script reading
    ``working_set.carries_files`` next to ``transferred: true`` took for a
    contradiction. This function is only reached once the files were sent, so the
    block says it carried them, under ``sent_digest`` — the carrying digest the
    gate bound (the consent's ``confirm``) — when the caller has it.
    """
    out = dict(planned)
    out["side"] = "source"
    out["carries_files"] = True
    if sent_digest:
        out["scope_digest"] = sent_digest
    if not isinstance(outcome, Mapping) or outcome.get("transferred") is not True:
        out["state"] = "outcome_unknown"
        out.pop("transferred", None)
        return out
    from local_operator.network import workset_text

    by_path: dict[str, Mapping[str, Any]] = {}
    for item in outcome.get("worktrees") or []:
        if isinstance(item, Mapping):
            by_path.setdefault(str(item.get("path") or ""), item)
    rows = []
    for raw in out.get("worktrees") or []:
        if not isinstance(raw, Mapping):
            continue
        row = {k: v for k, v in raw.items() if k not in ("reason",)}
        if raw.get("reason"):
            row["reason"] = raw["reason"]
        hit = by_path.get(str(raw.get("path") or ""))
        state = str(hit.get("state") or "") if hit is not None else ""
        if state in OUTCOME_ROW_STATES:
            row["state"] = state
            reason = str(hit.get("reason") or "") if hit is not None else ""
            row.pop("reason", None)
            if state == "left_behind":
                row["reason"] = reason if workset_text.known_reason(reason) else "not_transferred"
        else:
            # Our row, unnamed by the answer (or a state outside the closed set): the
            # destination did not say what became of it. Not "stayed" — this
            # block's own pre-invite state predates the prepare that sent the files.
            row["state"] = "outcome_unknown"
            row.pop("reason", None)
        rows.append(row)
    out["worktrees"] = rows
    out["transferred"] = True
    out["state"] = "transferred"
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
