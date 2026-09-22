"""The network store: records, secrets, invites and outbox queues on disk.

ONE WRITER, ONE PLACE, AND ONE LOCK PER TARGET. Everything under
``<config>/network`` is written here, by ``_write_private_json`` and
``_write_private_text`` — staged under a name unique to the process, the thread
and the call, chmod 0600, ``os.replace`` — so a reader sees either the old file or
the new one and never a torn one, and the writers of ONE file are serialised
against each other by :func:`_write_lock`. The guarantee is stated at the strength
the repository's other staged write already documents: it is durability of
*process*, not of *host* (no fsync of the directory, so a rename outlives a crashed
process but not a power cut).

THE LOCK SERIALISES WRITES, AND A CALLER'S OWN READ-MODIFY-WRITE NEEDS THE SAME
LOCK TOO. Both of the guarantees above are about ONE write: two callers of
``save`` cannot tear the file or mint the same sequence, and neither of those
covers the wider window a caller opens with ``load`` → mutate → ``save``. A record
materialised before another writer's ``save`` still carries the pre-write
membership, epoch and tombstones, and writing it back REVERTS them — the same
loss the sequence fix closed, one level up, and silent for the same reason
(nothing on the read path can tell a stale write from a fresh one).
:func:`mutate` is the whole of the answer: it holds this target's lock and reads
INSIDE it, so the record a body edits cannot predate a write it never saw.

The lock, and therefore :func:`mutate`, is IN-PROCESS — the same boundary the
write lock has always had, because the writers are several threads of one relay
process. TWO PROCESSES writing one record are outside it, and ONE such writer is
still open rather than merely hypothetical: ``network/cli.py:_cmd_rename`` saves
without asking a live relay first, where every other mutating verb routes through
``_relay_call`` and writes locally only when nothing answers. Closing that one
needs a lock the filesystem holds (``flock``), not this one, so it is named here
rather than left to be rediscovered.

THE SECRET LIVES IN A SEPARATE FILE FROM THE RECORD, deliberately. The record is
what ``lop network show --json`` dumps, what a future syncer copies, and what the
control socket returns; keeping key material out of it means there is no
redaction step for a surface to forget. That is the same inversion the secret
store enforces ("no surface returns a value to the model").

A CORRUPT RECORD IS QUARANTINED, NEVER DELETED. ``<network_id>.json.corrupt``
plus a warning, because a membership list is the one file where a mistake must
not be quietly reinterpreted: a record that failed to parse and was then
replaced would silently forget who is a member.

THE ``run/peers`` ACCESSORS ARE FUNCTION-LOCAL IMPORTS of
``session.runtime.registry``: this module is imported by ``network/cli.py`` on the
CLI startup path, and the registry drags ``procstate`` (which can shell out to
``ps``) behind it. A ``lop --version`` should not pay for that.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from secrets import token_hex
from typing import Any, Callable, Iterator, TextIO

from local_operator.network.identity import network_root
from local_operator.network.types import (
    PEERS_RUN_DIRNAME,
    MemberRecord,
    MeshRefusal,
    NetworkRecord,
    PairDecision,
    PeerRecord,
    PendingPairing,
    SecretState,
)

FILE_MODE = 0o600

NETWORKS_DIRNAME = "networks"
OUTBOX_DIRNAME = "outbox"
#: Where a pairing waits for the inviter's human (see :func:`pending_dir`).
PENDING_DIRNAME = "pending"
CATALOG_FILENAME = "catalog.json"
AUDIT_FILENAME = "audit.jsonl"
CORRUPT_SUFFIX = ".corrupt"

#: Invite entries are kept a day — long past any plausible pairing, short enough
#: that a burned token's record does not accumulate forever.
INVITE_PRUNE_AGE_S = 24 * 60 * 60.0

#: A tombstone's ROW is pruned after this long, but its id stays in
#: ``removed_ids`` forever: the id is what prevents a re-admission, and a
#: membership list that grew forever is a file someone eventually edits by hand.
TOMBSTONE_PRUNE_AGE_S = 90 * 24 * 60 * 60.0


def new_network_id() -> str:
    """``"n_" + 24 hex chars`` — random, not secret, stable, and a safe filename."""
    return f"n_{token_hex(12)}"


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def networks_dir(root: Path | None = None) -> Path:
    path = network_root(root) / NETWORKS_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def record_path(network_id: str, root: Path | None = None) -> Path:
    return networks_dir(root) / f"{network_id}.json"


def secrets_path(network_id: str, root: Path | None = None) -> Path:
    return networks_dir(root) / f"{network_id}.secrets.json"


def outbox_dir(root: Path | None = None) -> Path:
    path = network_root(root) / OUTBOX_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def peer_outbox_dir(device_id: str, root: Path | None = None) -> Path:
    """The durable per-peer queue for membership traffic.

    A subdirectory per peer, because the queue is drained by device and a flat
    directory of frames would need every file read to answer "what is queued for
    d_…" — and the answer is needed on every reconnect.
    """
    path = outbox_dir(root) / device_id
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def invite_path(invite_id: str, root: Path | None = None) -> Path:
    """Where a minted token is written: ``<config>/network/outbox/<id>.invite``."""
    return outbox_dir(root) / f"{invite_id}.invite"


def catalog_path(root: Path | None = None) -> Path:
    return network_root(root) / CATALOG_FILENAME


def audit_path(root: Path | None = None) -> Path:
    return network_root(root) / AUDIT_FILENAME


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


@dataclass
class _WriteLock:
    """One target's lock, and how many threads are holding or waiting for it."""

    lock: threading.RLock
    users: int = 0


#: ``{target: _WriteLock}``. Module state rather than a per-file one, because the
#: writer is a plain function the relay calls from several threads.
_WRITE_LOCKS: dict[str, _WriteLock] = {}
_WRITE_LOCKS_GUARD = threading.Lock()


@contextmanager
def _write_lock(target: Path) -> Iterator[None]:
    """Serialise the writers of ONE target, within this process.

    WHY A LOCK AS WELL AS A UNIQUE STAGING NAME, and why the two faults QA round
    15 filed are ONE fault. ``_write_private_json`` staged through
    ``.{name}.{pid}.tmp`` and ``os.getpid()`` is not unique inside a process, so
    two THREADS writing the same record shared one staging file: one thread's
    ``os.replace`` (or the ``finally`` unlink) could take the file away under the
    other, which then died at ``os.chmod`` with ``FileNotFoundError`` — and the
    body that *did* land was whichever thread last wrote into the shared file,
    not whichever last called ``save`` (Q15-2's ``device_name`` reading
    ``device-b`` after the name had been set to ``pixel-8``). The same window let
    the record's ``sequence`` run BACKWARDS, because ``save`` bumped whichever
    in-memory copy its caller held: two writers that had read the same file both
    wrote the same next number, and the update that lost was invisible (~15% of
    record writes in QA's sweep).

    ``relay.RelayServer`` serves in-process while the same process's CLI and
    session path write the same record (``sync_self_endpoints`` beside
    ``store.save`` from a session thread is the shape QA measured), so threads are
    the whole of the reachable concurrency here and a thread lock is what closes
    it. The lock is keyed on the target path as this process spells it — not on
    ``realpath``, which would spend a syscall on the hot path for callers that all
    build their path from the same root through :func:`record_path`.

    An entry is dropped once its last user leaves, because the outbox queues write
    one file PER FRAME: a registry that kept an entry per target ever written would
    grow with every frame queued. ``users`` is incremented before the acquire, so a
    waiting thread keeps its entry alive and no two threads can ever hold two
    different locks for one target.
    """
    key = os.fspath(target)
    with _WRITE_LOCKS_GUARD:
        entry = _WRITE_LOCKS.get(key)
        if entry is None:
            entry = _WriteLock(threading.RLock())
            _WRITE_LOCKS[key] = entry
        entry.users += 1
    entry.lock.acquire()
    try:
        yield
    finally:
        entry.lock.release()
        with _WRITE_LOCKS_GUARD:
            entry.users -= 1
            if entry.users == 0:
                del _WRITE_LOCKS[key]


def _stage_private(target: Path, emit: Callable[[TextIO], None]) -> Path:
    """Stage beside ``target``, 0600, and rename it into place.

    THE CALLER HOLDS THIS TARGET'S WRITE LOCK (see :func:`_write_lock`), which is
    why this is not a second entry point: the staging name is unique to the
    process, the thread AND the call, so it cannot be shared by two writers even
    if a future caller forgets the lock — the suffix is belt to the lock's braces,
    not a substitute for it.
    """
    directory = target.parent
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    temporary = directory / (
        f".{target.name}.{os.getpid()}.{threading.get_ident()}.{token_hex(4)}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            emit(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, FILE_MODE)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
    return target


def _write_private_json(target: Path, payload: Any) -> Path:
    """Staged 0600 JSON write, then rename. One of this package's two write paths."""

    def emit(handle: TextIO) -> None:
        json.dump(payload, handle, ensure_ascii=False)

    with _write_lock(target):
        return _stage_private(target, emit)


def _write_private_text(target: Path, text: str) -> Path:
    """Staged 0600 text write, then rename — the invite token's path.

    WHY NOT ``Path.write_text``, WHICH IS WHAT THIS REPLACED: that creates the file
    at the umask's mode and chmods it afterwards, so a BEARER CREDENTIAL sat
    readable to every local user for the length of the write while a reader could
    see a truncated token. Staging it gives the file 0600 in the same instant it
    becomes visible at the target, and nothing partial ever is, which is the deal
    every other file in this package already had.
    """

    def emit(handle: TextIO) -> None:
        handle.write(text)

    with _write_lock(target):
        return _stage_private(target, emit)


def _sequence_on_disk(target: Path) -> int:
    """The sequence the record file carries right now, or 0. Never raises."""
    data = _read_json(target)
    if data is None:
        return 0
    try:
        return int(data.get("sequence") or 0)
    except (TypeError, ValueError):
        return 0


def save(record: NetworkRecord, root: Path | None = None) -> Path:
    """Write the network record. Never carries key material (see the module docstring).

    THE SEQUENCE IS TAKEN FROM THE FILE, UNDER THE LOCK, not from the caller's
    copy. ``record.sequence`` describes the snapshot the caller loaded, and a
    caller's snapshot is stale by construction — the relay holds one while the
    CLI writes — so the bump has to happen against the file the write is about to
    replace or two writers mint the same number and one update is lost silently.
    ``max`` against the caller's own copy is deliberate: :func:`_sequence_on_disk`
    answers 0 for a record it cannot parse, and a torn read must never license a
    number LOWER than the caller already held, because every peer's "do I already
    have this?" compares sequences.
    """
    target = record_path(record.network_id, root)
    with _write_lock(target):
        record.sequence = max(record.sequence, _sequence_on_disk(target)) + 1
        payload = record.to_json()

        def emit(handle: TextIO) -> None:
            json.dump(payload, handle, ensure_ascii=False)

        return _stage_private(target, emit)


@contextmanager
def mutate(network_id: str, root: Path | None = None) -> Iterator[NetworkRecord]:
    """Read-modify-write ONE record, holding its write lock across the whole span.

    THE FAULT THIS CLOSES, in the words of the site that had it: the pair
    listener read a record, asked a HUMAN to compare a code, and then wrote the
    admission back — minutes later, from the snapshot it had read (relay.py,
    ``_run_pair_listener``). Anything another thread wrote in the meantime was
    reverted by that write, and the things in flight there are exactly the ones
    that must not be lost: a peer's epoch rotation (members, epoch, the secret
    that goes with it) and a membership pull (a newly admitted member). Nothing
    reported it, because the file that landed was a well-formed record with a
    HIGHER sequence — the write was newer than the one it clobbered, and only its
    CONTENT was older.

    So the rule is: the record a body edits is read AFTER this lock is taken, and
    the body's ``save`` happens before the lock is released. Two bodies therefore
    run one after the other — the second reads what the first wrote — and no
    caller can hold a pre-write snapshot across another's write.

    THE HELPER DOES NOT WRITE ON ITS OWN, and that is deliberate: the body calls
    :func:`save` when it changed something, exactly as it did before. A body that
    RAISES after mutating has to decide whether its change must survive the
    exception (the join path writes a consumed invite and re-raises on purpose),
    and a write-on-exit could not make that decision for it. Passing the SAME
    ``root`` the body passes to ``save`` matters: the lock is keyed on the target
    path as this process spells it, so a different spelling is a different lock and
    the nested write is no longer serialised against the outer one.

    KEEP THE BODY TIGHT. It holds a lock other threads want — the relay's
    heartbeat and membership loops write this same record — so a body that waits on
    a peer, a human or a clock belongs OUTSIDE the block, with the read RE-VALIDATED
    inside it (the join path's own rule: "the record on disk is the authority").

    Raises ``FileNotFoundError`` when the record is not there, like :func:`load`.
    """
    target = record_path(network_id, root)
    with _write_lock(target):
        yield load(network_id, root)


def save_secrets(state: SecretState, root: Path | None = None) -> Path:
    return _write_private_json(secrets_path(state.network_id, root), state.to_json())


def save_invite_token(invite_id: str, token: str, root: Path | None = None) -> Path:
    """Write a minted invite token to its own 0600 file.

    The token is a BEARER CREDENTIAL, so it goes to a file and never to stdout:
    a token printed by a command ends up in the agent's transcript, and the
    transcript is replayed to the provider on every later turn.
    """
    return _write_private_text(invite_path(invite_id, root), token + "\n")


# ---------------------------------------------------------------------------
# Pairing confirmation: the pending record and the human's decision
# ---------------------------------------------------------------------------


def pending_dir(root: Path | None = None) -> Path:
    """Where a pairing waits for a human: ``<config>/network/pending``.

    Its own directory rather than the outbox, because its lifetime is seconds and
    a reader must be able to answer "is a pairing waiting?" without scanning
    queue files.
    """
    path = network_root(root) / PENDING_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def pending_path(invite_id: str, root: Path | None = None) -> Path:
    return pending_dir(root) / f"{invite_id}.pending.json"


def decision_path(invite_id: str, root: Path | None = None) -> Path:
    return pending_dir(root) / f"{invite_id}.decision.json"


def save_pending_pairing(pending: PendingPairing, root: Path | None = None) -> Path:
    """Write the waiting pairing, 0600. Deleted with its decision."""
    return _write_private_json(pending_path(pending.invite_id, root), pending.to_json())


def pending_pairing(invite_id: str, root: Path | None = None) -> PendingPairing | None:
    """The waiting pairing, or ``None`` when there is none (expired included).

    ``None`` rather than an exception: "nobody is pairing right now" is the
    ordinary state of a healthy device, and every caller has a sentence for it.
    """
    data = _read_json(pending_path(invite_id, root))
    if data is None:
        return None
    try:
        return PendingPairing.from_json(data)
    except (TypeError, ValueError):
        _quarantine(pending_path(invite_id, root))
        return None


def pending_pairings(root: Path | None = None, *, now: float | None = None) -> list[PendingPairing]:
    """Every OPEN pairing, oldest first. Expired ones are cleared as they are found."""
    moment = time.time() if now is None else now
    found: list[PendingPairing] = []
    for path in sorted(pending_dir(root).glob("*.pending.json")):
        invite_id = path.name[: -len(".pending.json")]
        pending = pending_pairing(invite_id, root)
        if pending is None:
            continue
        if not pending.is_open(moment):
            clear_pending_pairing(invite_id, root)
            clear_pair_decision(invite_id, root)
            continue
        found.append(pending)
    return sorted(found, key=lambda row: row.issued_at)


def clear_pending_pairing(invite_id: str, root: Path | None = None) -> None:
    pending_path(invite_id, root).unlink(missing_ok=True)


def save_pair_decision(decision: PairDecision, root: Path | None = None) -> Path:
    """Write the human's answer where the relay's pairing loop will see it."""
    return _write_private_json(decision_path(decision.invite_id, root), decision.to_json())


def pair_decision(invite_id: str, root: Path | None = None) -> PairDecision | None:
    data = _read_json(decision_path(invite_id, root))
    if data is None:
        return None
    try:
        return PairDecision.from_json(data)
    except (TypeError, ValueError):
        _quarantine(decision_path(invite_id, root))
        return None


def clear_pair_decision(invite_id: str, root: Path | None = None) -> None:
    decision_path(invite_id, root).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Purge: what `lop network uninstall --purge[-identity]` deletes
# ---------------------------------------------------------------------------


def purge_network_artifacts(
    network_ids: list[str] | None = None, root: Path | None = None
) -> dict[str, Any]:
    """Forget the network records, invites, queues, pending pairings, and — when
    EVERY network goes — the audit log too.

    THE SCOPE IS THE WHOLE POINT (design §6, §12): this is what makes a device
    forget networks it can no longer reach, and it deliberately does NOT touch the
    device identity keypair, which other networks address this device by.

    The audit log is ONE file per install that records ``network_id`` per entry,
    so a per-network purge cannot delete part of it without rewriting a forensic
    log — which is the one thing a forensic log may not be. It is therefore
    deleted only when this call covers every network the device knows, and the
    receipt says which of the two happened.
    """
    known = list_networks(root)
    targets = {record.network_id for record in known} if network_ids is None else set(network_ids)
    selected = [record for record in known if record.network_id in targets]
    removed: dict[str, Any] = {
        "networks": [],
        "invites": 0,
        "queues": 0,
        "pending": 0,
        "audit_files": [],
        "audit_kept": False,
        "catalog": False,
    }
    for record in selected:
        removed["networks"].append(f"{record.name} ({record.network_id})")
        for invite in record.invites:
            invite_file = invite_path(invite.invite_id, root)
            if invite_file.exists():
                invite_file.unlink()
                removed["invites"] += 1
            for helper in (pending_path, decision_path):
                if helper(invite.invite_id, root).exists():
                    helper(invite.invite_id, root).unlink()
                    removed["pending"] += 1
        for member in record.members:
            queue = outbox_dir(root) / member.device_id
            if queue.is_dir():
                shutil.rmtree(queue, ignore_errors=True)
                removed["queues"] += 1
        record_path(record.network_id, root).unlink(missing_ok=True)
        secrets_path(record.network_id, root).unlink(missing_ok=True)

    covers_everything = bool(known) and targets >= {record.network_id for record in known}
    if covers_everything:
        # Every queue goes, not only the ones the surviving member rows name: a queue
        # whose device has left and whose row has been pruned (90 days) belongs to no
        # network this record still mentions, and an uninstall that leaves frames for
        # a device it no longer knows is the kind of leftover a purge exists to
        # prevent. A SCOPED purge cannot attribute those, so it keeps them and says so.
        for queue in sorted(outbox_dir(root).iterdir()):
            if queue.is_dir():
                shutil.rmtree(queue, ignore_errors=True)
                removed["queues"] += 1
        for token in sorted(outbox_dir(root).glob("*.invite")):
            token.unlink()
            removed["invites"] += 1
        for suffix in [""] + [f".{index}.gz" for index in range(100)]:
            candidate = audit_path(root).with_name(AUDIT_FILENAME + suffix)
            if not candidate.exists():
                continue
            candidate.unlink()
            removed["audit_files"].append(candidate.name)
        if catalog_path(root).exists():
            catalog_path(root).unlink()
            removed["catalog"] = True
    else:
        removed["audit_kept"] = True
    # Pairings parked for a purged network, even when the invite row they name was
    # pruned: their file carries `network_id`, so the attribution is exact rather
    # than a guess from a filename.
    for pending_file in sorted(pending_dir(root).glob("*.pending.json")):
        row = pending_pairing(pending_file.name[: -len(".pending.json")], root)
        if row is None or row.network_id not in targets:
            continue
        pending_file.unlink()
        clear_pair_decision(row.invite_id, root)
        removed["pending"] += 1
    return removed


def purge_identity(root: Path | None = None) -> list[str]:
    """Delete this device's keypair. The unrecoverable act.

    Separate from :func:`purge_network_artifacts` because the two have different
    blast radii: this key is what every OTHER network addresses this device by, so
    the caller must have a human's named confirmation before calling it.

    The file, then the directory: ``identity/device.json`` is the only content the
    identity directory ever holds, so leaving the empty directory behind would
    leave `lop network status` reporting an identity path with nothing in it.
    """
    from local_operator.network.identity import identity_dir, identity_path

    deleted: list[str] = []
    keypair = identity_path(root)
    if keypair.exists():
        keypair.unlink()
        deleted.append(keypair.name)
    directory = identity_dir(root)
    if directory.is_dir() and not any(directory.iterdir()):
        directory.rmdir()
        deleted.append("identity/")
    return deleted


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def load(network_id: str, root: Path | None = None) -> NetworkRecord:
    path = record_path(network_id, root)
    data = _read_json(path)
    if data is None:
        raise FileNotFoundError(f"no network record at {path}")
    return NetworkRecord.from_json(data)


def load_secrets(network_id: str, root: Path | None = None) -> SecretState:
    path = secrets_path(network_id, root)
    data = _read_json(path)
    if data is None:
        raise FileNotFoundError(f"no epoch secrets at {path}")
    state = SecretState.from_json(data)
    if not state.secret:
        raise MeshRefusal(
            "secrets_missing",
            f"the epoch secrets at {path} carry no current key; this network cannot "
            "authenticate a link until it is re-created",
        )
    return state


def read_config(path: tuple[str, ...], default: Any, root: Path | None = None) -> Any:
    """One nested ``network.*`` value from the config store, or ``default``.

    THE ONE READER for this package's configuration. ``relay.NetworkSettings``
    and ``audit.AuditLog.from_config`` both go through it, because two spellings
    of "where a network setting lives" is how a key ends up read by one consumer
    and silently ignored by the other — the failure #576's reaper toggle spent
    weeks being. ``ConfigManager.get_nested_value`` walks the same tuple
    ``settings_io`` writes, so the registry entry and this read cannot disagree
    about the path.

    Function-local import of ``ConfigManager``: it pulls the YAML store, and this
    module is on the CLI startup path.
    """
    from local_operator.config import ConfigManager
    from local_operator.paths import config_dir

    manager = ConfigManager(root or config_dir())
    return manager.get_nested_value(path, default)


def require_secrets(network_id: str, root: Path | None = None) -> SecretState:
    """The epoch secrets, or a NAMED refusal when this device no longer has them.

    A missing secrets file is a legitimate state, not a bug: ``lop network
    disconnect`` deletes it on purpose. Every operator-facing reader goes through
    this so the answer is a sentence ("this device left that network; re-pair
    with an invite") and a code, never a ``FileNotFoundError`` traceback — the
    rule the CLI's whole error surface now follows.
    """
    path = secrets_path(network_id, root)
    if not path.exists():
        raise MeshRefusal(
            "no_network_secret",
            "this device has no secret for that network: it was forgotten with "
            "`lop network rm`, deleted by `lop network disconnect`, or never written. "
            "Rejoin with `lop network join` and a fresh invite from a member.",
        )
    return load_secrets(network_id, root)


def list_networks(root: Path | None = None) -> list[NetworkRecord]:
    """Every readable network record, quarantining (never deleting) the rest.

    Sorted by name then id so a listing is stable, and the sort is here rather
    than at each surface: two surfaces that disagreed about order would look like
    two different networks to someone comparing screens.
    """
    records: list[NetworkRecord] = []
    for path in sorted(networks_dir(root).glob("*.json")):
        if path.name.endswith(".secrets.json") or path.name.endswith(CORRUPT_SUFFIX):
            continue
        data = _read_json(path)
        if data is None:
            _quarantine(path)
            continue
        try:
            records.append(NetworkRecord.from_json(data))
        except (TypeError, ValueError):
            _quarantine(path)
    return sorted(records, key=lambda record: (record.name, record.network_id))


def match_networks(records: list[NetworkRecord], target: str) -> list[NetworkRecord]:
    """Networks a ``<network>`` argument could mean, id first then name.

    Names are NOT unique by design (§16 Q8: two networks may share a display
    name), so an ambiguous argument returns every candidate and the CLI refuses
    rather than picking one — a rename must never have to rewrite member records,
    and a `rm` must never be able to hit the wrong network because two were
    called "home".
    """
    exact = [record for record in records if record.network_id == target]
    if exact:
        return exact
    return [record for record in records if record.name == target]


def forget(network_id: str, root: Path | None = None) -> list[str]:
    """Remove this device's own record, secrets and queued frames for a network.

    Purely local (``lop network rm``), and it is the documented remedy for a
    device whose peers have refused it: it is what an operator runs on the device
    that was removed. Deliberately does NOT touch the audit log — the trail of
    what happened outlives the membership.
    """
    removed: list[str] = []
    for path in (record_path(network_id, root), secrets_path(network_id, root)):
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _quarantine(path: Path) -> None:
    """Move an unparsable record aside so the next run can still list the others."""
    target = path.with_name(path.name + CORRUPT_SUFFIX)
    try:
        os.replace(path, target)
    except OSError:
        return
    print(
        f"warning: {path.name} could not be parsed and was set aside as {target.name}; "
        "the network it described is no longer listed",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Prune
# ---------------------------------------------------------------------------


def prune(record: NetworkRecord, *, now: float | None = None) -> dict[str, int]:
    """Drop what ages out, keeping what must never be forgotten.

    Two distinct rules, and conflating them is the mistake this function exists to
    avoid:

    * an INVITE entry older than a day goes entirely — its token is long dead;
    * a TOMBSTONE's row goes after 90 days, but the id stays in ``removed_ids``
      forever, because the id is the thing that prevents a re-admission. Pruning
      the list instead of the row would un-burn every removed device.
    """
    moment = time.time() if now is None else now
    invites_before = len(record.invites)
    record.invites = [
        invite
        for invite in record.invites
        if moment - invite.minted_at <= INVITE_PRUNE_AGE_S
        or (invite.redeemed_at is not None and moment - invite.redeemed_at <= INVITE_PRUNE_AGE_S)
    ]
    tombstones_before = len(record.members)
    kept: list[MemberRecord] = []
    for member in record.members:
        if member.removed_at is not None and moment - member.removed_at > TOMBSTONE_PRUNE_AGE_S:
            continue
        kept.append(member)
    record.members = kept
    record.pending = [
        entry
        for entry in record.pending
        if moment - float(entry.get("started_at") or moment) <= INVITE_PRUNE_AGE_S
    ]
    return {
        "invites_dropped": invites_before - len(record.invites),
        "tombstone_rows_dropped": tombstones_before - len(record.members),
        "burned_ids_kept": len(record.removed_ids),
    }


# ---------------------------------------------------------------------------
# The run/peers namespace (A2)
# ---------------------------------------------------------------------------


def run_record_path(pid: int, root: Path | None = None) -> Path:
    """Where one relay's discovery record lives: ``run/peers/<pid>.json``."""
    from local_operator.session.runtime import registry

    return registry.record_path(pid, root, dirname=PEERS_RUN_DIRNAME)


def publish_peer_record(record: PeerRecord, root: Path | None = None) -> Path:
    """Publish (or heartbeat) the relay's record through the shared publication path.

    Delegating to ``session.runtime.registry`` is the whole point of A2's
    consequence list: the atomic staged write, the 0600-and-rename, the heartbeat
    and the stale-record reaping are the SAME code that session records use, and a
    second copy of them here would be free to disagree about what "alive" means.
    """
    from local_operator.session.runtime import registry

    return registry.publish(record, root, dirname=PEERS_RUN_DIRNAME)


def unpublish_peer_record(pid: int, root: Path | None = None) -> None:
    from local_operator.session.runtime import registry

    registry.unpublish(pid, root, dirname=PEERS_RUN_DIRNAME)


def scan_peer_records(
    root: Path | None = None, *, reap: bool = True
) -> list[tuple[PeerRecord, str]]:
    """Every relay record, each with its ``live``/``wedged``/``stale`` verdict."""
    from local_operator.session.runtime import registry

    return registry.scan(root, dirname=PEERS_RUN_DIRNAME, parse=PeerRecord.from_json, reap=reap)


def peer_records(root: Path | None = None, *, live_only: bool = True) -> list[PeerRecord]:
    """The relays this device can see, newest-first is irrelevant: pid order is stable."""
    records: list[PeerRecord] = []
    for record, state in scan_peer_records(root):
        if live_only and state != "live":
            continue
        records.append(record)
    return records


def find_own_relay(root: Path | None = None) -> PeerRecord | None:
    """The live relay record for this install, or ``None`` when none is running.

    One relay per install, so a live scan yields at most one record; the first is
    returned rather than raising on the overlap window after a restart, when a
    reaped-but-not-yet-removed record can still be present. Callers dial the
    record's ``control_port`` with its ``control_key`` — the CLI is a different
    process, which is exactly why the record carries those two fields.

    THIS ANSWERS "IS THERE A RELAY TO TALK TO", NOT "IS A RELAY RUNNING". A
    record whose owner has stopped heartbeating is deliberately excluded here,
    because every caller is about to dial the control socket and would only hang.
    A DIAGNOSTIC asks the other question and must use :func:`scan_own_relay`:
    reporting a wedged relay as an absent one beside a payload that carries its pid
    is a diagnostic contradicting itself (QA round 3, Q-R3-4).
    """
    records = peer_records(root)
    return records[0] if records else None


def scan_own_relay(root: Path | None = None) -> tuple[PeerRecord | None, str]:
    """This install's relay record with the classifier's verdict for it.

    Returns ``(record, state)`` where ``state`` is ``registry.classify``'s word —
    ``live`` (pid alive, heartbeat fresh) or ``wedged`` (pid alive, the owner has
    not reported inside ``HEARTBEAT_TIMEOUT_S``; NOT proof the process is dead) —
    and ``(None, "")`` when this install has no relay process at all. A record
    whose pid is gone is reaped by the underlying scan and is not a relay.

    The record and the verdict travel TOGETHER, so a caller cannot print a pid
    without also holding the answer to "and is it answering", which is the one way
    a status payload came to report ``relay_running: false`` next to a live
    ``record.pid``. A caller that wants to DIAL the relay still wants
    :func:`find_own_relay`; this one is for describing the machine.
    """
    for record, state in scan_peer_records(root):
        if state in ("live", "wedged"):
            return record, state
    return None, ""


# ---------------------------------------------------------------------------
# The durable per-peer outbox
# ---------------------------------------------------------------------------


def enqueue_frame(
    device_id: str,
    frame: dict[str, Any],
    *,
    removed: bool,
    root: Path | None = None,
    now: float | None = None,
) -> Path:
    """Queue one membership frame for a peer that is offline.

    THE CONVERGENCE RULE, ENFORCED AT THE WRITER: a frame that carries the network
    secret is never persisted for a REMOVED recipient. The queue is a file on
    disk, and a file is the one place a secret outlives the decision to stop
    trusting a device — so the check lives here rather than at each enqueueing
    call site, where the next caller would have to remember it.

    ``removed`` is the caller's answer to "is this device a tombstone (or absent
    from the current member list) right now", passed in rather than looked up so
    that the writer cannot silently disagree with the membership code that just
    made the decision.
    """
    if removed and "secret" in frame:
        raise MeshRefusal(
            "removed_recipient",
            f"refusing to queue a rotation that carries this network's secret for {device_id}: "
            "that device is not a member any more",
        )
    directory = peer_outbox_dir(device_id, root)
    moment = time.time() if now is None else now
    name = f"{int(moment * 1000):013d}-{os.getpid()}-{token_hex(4)}.frame"
    return _write_private_json(directory / name, frame)


def queued_frames(device_id: str, root: Path | None = None) -> list[tuple[Path, dict[str, Any]]]:
    """Queued frames in the order they were written (the filename is the clock).

    An unparsable queued frame is DELETED rather than quarantined: unlike a network
    record it carries no membership decision, and a queue that a torn file can
    block forever is a queue that never drains.
    """
    directory = peer_outbox_dir(device_id, root)
    queued: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(directory.glob("*.frame")):
        data = _read_json(path)
        if data is None:
            path.unlink(missing_ok=True)
            continue
        queued.append((path, data))
    return queued


def drop_frame(path: Path) -> None:
    path.unlink(missing_ok=True)
