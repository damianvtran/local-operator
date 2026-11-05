"""Bounded authored-definition transactions, shared by every supported writer.

Agent metadata and its prompt are separate canonical files. An instance-local
lock cannot protect their CAS/write window from another process, nor can it
recover a process death between those files. This sidecar lock and write-ahead
journal add that boundary without migrating the canonical layout. State writers
share the boundary too, but never replay stale authored fields during autosave.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

from local_operator.hub_sync.definition import DefinitionRevisionMismatch

P = ParamSpec("P")
R = TypeVar("R")
_LOCKS_GUARD = threading.Lock()
_LOCKS: dict[str, threading.RLock] = {}
_LOCAL = threading.local()

#: Identities consumed IN THIS PROCESS, after the runtime server's
#: ``_live_challenges`` pattern: a leaked one-shot token is cleared on use, and
#: the store is pruned on access rather than by a timer. The journal on disk is
#: same-uid editable, so a rollback to ``prepared`` would otherwise re-apply an
#: operation that already landed here. Process-local and bounded ON PURPOSE: it
#: is a defence-in-depth guard for the threat model the architect recorded (an
#: ordinary session runs an unconfined shell), never a cross-process boundary.
_CONSUMED: dict[str, float] = {}
_CONSUMED_LOCK = threading.Lock()
_CONSUMED_TTL_S = 24 * 3600.0


def _consumed_key(root: Path, transaction_id: str) -> str:
    return str(Path(root).resolve()) + "|" + str(transaction_id)


def _consumed_error() -> "DefinitionTransactionConflict":
    return DefinitionTransactionConflict(
        "this operation was already applied in this process; never replay it — "
        "ask the person for the change again"
    )


def _raise_if_consumed(key: str) -> None:
    with _CONSUMED_LOCK:
        moment = time.monotonic()
        for stale in [name for name, seen in _CONSUMED.items() if moment - seen > _CONSUMED_TTL_S]:
            _CONSUMED.pop(stale, None)
        if key in _CONSUMED:
            raise _consumed_error()


def _mark_consumed(key: str) -> None:
    with _CONSUMED_LOCK:
        _CONSUMED[key] = time.monotonic()


class DefinitionTransactionConflict(DefinitionRevisionMismatch):
    """The definition moved or a journal needs explicit recovery."""


def commitment(value: Any) -> str:
    """A content token, not authority: callers must authenticate signed inputs."""
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(data.encode("utf-8")).hexdigest()


@contextmanager
def definition_lock(root: Path, *, timeout: float = 2.0) -> Iterator[None]:
    """One bounded, thread-reentrant cross-process lock for agents AND teams.

    Lock order is always this boundary before a registry's existing local/team
    persistence lock. Nested registry helpers reuse this thread's open handle,
    never flock a second descriptor and accidentally release another writer.
    """
    root = Path(root).resolve()
    key = str(root)
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(key, threading.RLock())
    deadline = time.monotonic() + max(0.0, timeout)
    if not lock.acquire(timeout=max(0.0, timeout)):
        raise DefinitionTransactionConflict("definition writer is busy; retry later")
    held = getattr(_LOCAL, "held", None)
    if held is None:
        held = _LOCAL.held = {}
    try:
        if key in held:
            yield
            return
        root.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(root / ".profile-definitions.lock", flags, 0o600)
        try:
            while True:
                try:
                    if os.name == "nt":  # pragma: no cover - Windows CI
                        import msvcrt

                        os.lseek(fd, 0, os.SEEK_SET)
                        if os.fstat(fd).st_size == 0:
                            os.write(fd, b"\0")
                        os.lseek(fd, 0, os.SEEK_SET)
                        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as exc:
                    if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                        raise
                    if time.monotonic() >= deadline:
                        raise DefinitionTransactionConflict(
                            "definition writer is busy; retry later"
                        ) from exc
                    time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
            held[key] = fd
            try:
                yield
            finally:
                held.pop(key, None)
        finally:
            # Closing releases exactly this descriptor's kernel lock, including
            # exception paths; no broad process/file cleanup is involved.
            os.close(fd)
    finally:
        lock.release()


def definition_writer(fn: Callable[P, R]) -> Callable[P, R]:
    """Serialize supported registry writes before their existing inner locks."""

    @wraps(fn)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        registry: Any = args[0]
        root = Path(registry.config_dir)
        with definition_lock(root):
            active = getattr(_LOCAL, "owners", {}).get(str(root.resolve()))
            if active is not None and active is not registry:
                # Two registry instances nested on one thread are distinct
                # writers, not reentrant helper calls. Refuse instead of
                # silently admitting an interleaving inside a transaction.
                raise DefinitionTransactionConflict(
                    "competing definition writer inside transaction"
                )
            if fn.__name__ == "create_agent":
                registry._refresh_agents_metadata()
            return fn(*args, **kwargs)

    return wrapped


def _durable_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    try:
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        if os.name != "nt":
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        temp.unlink(missing_ok=True)


def prepare(
    root: Path,
    *,
    transaction_id: str,
    before: Any,
    after: Any,
    read: Callable[[], Any] | None = None,
    binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write (or confirm) the UNCONSUMED record a later apply is checked against.

    Splitting preparation out is what makes ABSENCE unreadable as authority: a
    journal that is missing at apply time means this operation was never
    prepared here, so :func:`transact` refuses instead of treating a deleted
    crash record as a fresh grant. Idempotent for an identical unconsumed
    record; ``rolled_back`` may be prepared again only when the definition still
    matches the recorded ``before`` commitment, because that is the state the
    rollback restored.
    """
    _validate_identity(transaction_id)
    with definition_lock(root):
        path = Path(root) / "profile-transactions" / (transaction_id + ".json")
        key = _consumed_key(root, transaction_id)
        _raise_if_consumed(key)
        before_token, after_token = commitment(before), commitment(after)
        if not path.exists():
            if read is not None and commitment(read()) != before_token:
                raise DefinitionTransactionConflict("definition changed; nothing was written")
            record = {
                "schema": 1,
                "before": before_token,
                "after": after_token,
                "state": "prepared",
                "binding": binding,
            }
            _durable_json(path, record)
            return record
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("before") != before_token or previous.get("after") != after_token:
            raise DefinitionTransactionConflict("transaction identity has different images")
        if previous.get("binding") != binding:
            raise DefinitionTransactionConflict("transaction belongs to a different approval")
        if previous.get("state") == "prepared":
            return previous
        if previous.get("state") in {"complete", "applying", "failed"}:
            raise _consumed_error()
        if read is not None and commitment(read()) != before_token:
            raise DefinitionTransactionConflict("definition changed; nothing was written")
        previous["state"] = "prepared"
        _durable_json(path, previous)
        return previous


def _validate_identity(transaction_id: str) -> None:
    if (
        not transaction_id
        or len(transaction_id) > 128
        or any(
            char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for char in transaction_id
        )
    ):
        raise ValueError("invalid definition transaction identity")


def transact(
    root: Path,
    *,
    transaction_id: str,
    before: Any,
    after: Any,
    read: Callable[[], Any],
    write: Callable[[Any], None],
    backup: Callable[[Any], None] | None = None,
    binding: dict[str, Any] | None = None,
    one_shot: bool = False,
) -> dict[str, Any]:
    """CAS, backup, journal and mutation under ONE definition boundary.

    The journal is bookkeeping, never approval authority. A durable executor
    must pass authenticated before/after images again after restart. A journal
    matching neither image is a conflict, not an invitation to overwrite an
    intervening edit. An already-after image finishes bookkeeping without
    replaying mutation or backup. Ordinary exceptions roll back this operation;
    process death leaves the journal for explicit authenticated reconciliation.
    ``one_shot`` consumes durably before mutation; known faults are terminal by
    fresh-image proof, and an uncertain applying record can only finish an
    already-after image, never retry a before-looking (possibly ABA) row.
    """
    _validate_identity(transaction_id)
    with definition_lock(root):
        path = Path(root) / "profile-transactions" / (transaction_id + ".json")
        key = _consumed_key(root, transaction_id)
        _raise_if_consumed(key)
        before_token, after_token = commitment(before), commitment(after)
        # AN ABSENT JOURNAL IS NOT AUTHORISATION (architect review e9c4b99d):
        # deleting this same-uid file used to convert a crash record into a
        # silent fresh apply. Only ``prepare`` creates the record acted on here.
        if not path.exists():
            raise DefinitionTransactionConflict(
                "no prepared transaction for this operation; nothing was written — "
                "ask the person for the change again"
            )
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("before") != before_token or previous.get("after") != after_token:
            raise DefinitionTransactionConflict("transaction identity has different images")
        if previous.get("binding") != binding:
            raise DefinitionTransactionConflict("transaction belongs to a different approval")
        current = commitment(read())
        if previous.get("state") == "complete":
            # A later edit may deliberately restore the original bytes. That
            # does not revive a consumed operation's permission to write again.
            raise DefinitionTransactionConflict("completed transaction changed; never replay")
        if previous.get("state") != "prepared" and not (
            one_shot and previous.get("state") == "applying"
        ):
            raise DefinitionTransactionConflict(
                "this operation failed and rolled back; prepare it again — never replay it"
            )
        if one_shot and previous.get("state") == "applying" and current != after_token:
            # A crash between durable consumption and completion cannot prove
            # whether a before-looking row is untouched or a legitimate ABA.
            # Never reinterpret this uncertain operation as a fresh write.
            raise DefinitionTransactionConflict("consumed operation needs reconciliation")
        if current == after_token:
            result = {**previous, "state": "complete", "reconciled": True}
            _durable_json(path, result)
            _mark_consumed(key)
            return result
        if current != before_token:
            raise DefinitionTransactionConflict("definition changed; nothing was written")
        record = previous
        if backup is not None:
            backup(before)
        if one_shot:
            record["state"] = "applying"
            _durable_json(path, record)  # Consume BEFORE any authored mutation.
        try:
            write(after)
            if commitment(read()) != after_token:
                raise DefinitionTransactionConflict("definition write did not match its commitment")
        except Exception as primary:
            try:
                if one_shot:
                    # Storage adapters may wrap OSError. Classify from fresh bytes,
                    # never the exception's class or its potentially private text.
                    actual = commitment(read())
                    if actual != after_token and actual != before_token:
                        try:
                            write(before)
                        except Exception:
                            pass  # Readback below is the proof, not a successful call.
                        actual = commitment(read())
                    if actual == after_token:
                        record["state"] = "complete"
                    elif actual == before_token:
                        record["state"] = "failed"
                    else:
                        raise DefinitionTransactionConflict("write outcome requires reconciliation")
                    _durable_json(path, record)
                    _mark_consumed(key)
                    return record
                else:
                    # Ordinary saves retain rollback-and-raise semantics. This must
                    # share the handler above: a sibling except Exception would be
                    # unreachable and silently mark partial non-one-shot writes done.
                    # The same lock excludes competing writers during the rollback;
                    # failed recovery leaves prepared bookkeeping, never completion.
                    if commitment(read()) != before_token:
                        write(before)
                    if commitment(read()) != before_token:
                        raise DefinitionTransactionConflict(
                            "rollback did not restore its commitment; recovery required"
                        )
                    record["state"] = "rolled_back"
                    _durable_json(path, record)
                    raise
            except Exception as secondary:
                # An unreadable published image is UNKNOWN, not a new reason
                # to recover/restore or claim rollback. Keep the original save
                # fault outward; normal access owns recovery after the fault ends.
                if secondary is primary:
                    raise
                primary.add_note(
                    "definition outcome is unconfirmed; recovery required "
                    f"({type(secondary).__name__})"
                )
                raise primary from secondary

        record["state"] = "complete"
        _durable_json(path, record)
        _mark_consumed(key)
        return record


def agent_image(registry: Any, name: str) -> dict[str, Any] | None:
    """Fresh canonical metadata plus prompt, for a known local registry only."""
    registry._refresh_agents_metadata()
    row = registry.get_agent_by_name(name)
    if row is None:
        return None
    return {
        "row": row.model_dump(mode="json"),
        "instructions": registry.get_agent_system_prompt(row.id),
    }


def restore_agent_image(registry: Any, name: str, image: dict[str, Any] | None) -> None:
    """Restore an in-transaction snapshot; never resolve a caller-chosen root."""
    from local_operator.agents import AgentData

    current = registry.get_agent_by_name(name)
    if image is None:
        if current is not None:
            registry.delete_agent(current.id)
        return
    row = AgentData.model_validate(image["row"])
    if row.name != name or (current is not None and current.id != row.id):
        raise DefinitionTransactionConflict("snapshot identity does not match the target")
    registry.save_agent(row)
    registry.set_agent_system_prompt(row.id, str(image["instructions"]))


@contextmanager
def atomic_edit(
    root: Path,
    *,
    owner: Any,
    read: Callable[[], Any],
    restore: Callable[[Any], None],
    backup: Callable[[Any], None] | None = None,
    transaction_id: str | None = None,
) -> Iterator[None]:
    """Exception-atomic legacy edit; signed continuations use ``transact``.

    A legacy editor computes its after-image while validating registry fields,
    so it cannot supply a signed after commitment up front. Its before image
    is still journaled before any mutation and restored on ordinary failure.
    A process death leaves explicit ``prepared`` bookkeeping; no unsigned
    journal is ever treated as authority to replay or restore after restart.
    """
    with definition_lock(root):
        before = read()
        key = str(Path(root).resolve())
        active = getattr(_LOCAL, "owners", None)
        if active is None:
            active = _LOCAL.owners = {}
        if key in active:
            raise DefinitionTransactionConflict("nested definition transaction is not supported")
        edit_id = transaction_id or ("edit-" + uuid.uuid4().hex)
        _validate_identity(edit_id)
        consume_key = _consumed_key(root, edit_id)
        _raise_if_consumed(consume_key)
        path = Path(root) / "profile-transactions" / (edit_id + ".json")
        if transaction_id is not None and path.exists():
            # A NAMED edit is ONE operation, so a journal already on disk is that
            # operation's own earlier attempt: either it landed (complete), it was
            # restored (rolled_back), or it died mid-write (prepared, outcome
            # unknown). Only the restored case may run again; an unknown outcome
            # needs a fresh decision from the person, never a replay.
            seen = json.loads(path.read_text(encoding="utf-8"))
            if seen.get("state") == "rolled_back" and commitment(before) == seen.get("before"):
                path.unlink()
            elif seen.get("state") == "complete":
                raise _consumed_error()
            else:
                raise DefinitionTransactionConflict(
                    "an earlier attempt at this exact change was interrupted before it "
                    "finished, so its outcome is unknown; nothing was written — ask the "
                    "person for the change again"
                )
        record = {
            "schema": 1,
            "state": "prepared",
            "before": commitment(before),
            "before_image": before,
            "after": None,
        }
        # Install ownership only once preflight can no longer refuse: otherwise
        # a consumed/conflicting edit leaves a phantom nested transaction behind.
        active[key] = owner
        try:
            _durable_json(path, record)
            if backup is not None:
                backup(before)
            try:
                yield
            except Exception as primary:
                try:
                    # Read only the published row. Recovery here could rename a
                    # backup under the still-active fault and mask the save error.
                    if commitment(read()) != record["before"]:
                        restore(before)
                        if commitment(read()) != record["before"]:
                            raise DefinitionTransactionConflict(
                                "rollback did not restore its commitment; recovery required"
                            )
                    record["state"] = "rolled_back"
                    _durable_json(path, record)
                except Exception as secondary:
                    # Leave prepared/unknown bookkeeping intact unless readback
                    # proved rollback. Normal registry access owns later recovery.
                    primary.add_note(
                        "definition rollback is unconfirmed; recovery required "
                        f"({type(secondary).__name__})"
                    )
                    raise primary from secondary
                raise
            record.update(state="complete", after=commitment(read()))
            _durable_json(path, record)
            _mark_consumed(consume_key)
        finally:
            active.pop(key, None)


def team_image(registry: Any, team_id: str) -> dict[str, Any] | None:
    """Fresh team metadata AND briefs, under the shared definition boundary."""
    from local_operator.teams import read_team_snapshot

    row = read_team_snapshot(Path(registry.teams_dir), team_id)
    return None if row is None else row.model_dump(mode="json")


def team_definition_preflight(fn: Callable[P, R]) -> Callable[P, R]:
    """Reject detached invalid identities before either outer lock creates files."""

    @wraps(fn)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        from local_operator.teams import Team, validate_team_id

        target = (
            args[1]
            if len(args) > 1
            else kwargs["team" if fn.__name__ == "save_team" else "team_id"]
        )
        if fn.__name__ == "save_team":
            if not isinstance(target, Team):
                raise TypeError("save_team needs a Team")
            candidate = Team.model_validate(target.model_dump())
            validate_team_id(candidate.id)
        else:
            if not isinstance(target, str):
                raise TypeError("team identity must be a string")
            validate_team_id(target)
        return fn(*args, **kwargs)

    return wrapped


def team_definition_writer(fn: Callable[P, R]) -> Callable[P, R]:
    """Journal team updates/saves/deletes using the team's existing atomic swap.

    Unlike an agent's split metadata/prompt files, TeamRegistry already has a
    cross-process atomic directory-swap protocol. This outer journal gives both
    kinds the same operation-level before/after and rollback contract while
    preserving that existing storage and recovery implementation.
    """

    @wraps(fn)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        registry: Any = args[0]
        target = (
            args[1]
            if len(args) > 1
            else kwargs["team" if fn.__name__ == "save_team" else "team_id"]
        )
        root = Path(registry.config_dir)
        with definition_lock(root):
            if getattr(_LOCAL, "owners", {}).get(str(root.resolve())) is registry:
                return fn(*args, **kwargs)
            from local_operator.teams import Team

            if fn.__name__ == "save_team":
                if not isinstance(target, Team):
                    raise TypeError("save_team needs a Team")
                team_id = target.id
            else:
                team_id = str(target)

            def restore(image: Any) -> None:
                if image is None:
                    if team_image(registry, team_id) is not None:
                        registry.delete_team(team_id)
                else:
                    # save_team deliberately rehydrates briefs from disk to
                    # protect stale metadata-only saves. A transaction rollback
                    # is different: its captured briefs ARE authoritative.
                    with registry._persistence_lock():
                        registry._load()
                        registry._save_team_locked(
                            Team.model_validate(image), briefs_authoritative=True
                        )

            with atomic_edit(
                root, owner=registry, read=lambda: team_image(registry, team_id), restore=restore
            ):
                return fn(*args, **kwargs)

    return wrapped
