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


def transact(
    root: Path,
    *,
    transaction_id: str,
    before: Any,
    after: Any,
    read: Callable[[], Any],
    write: Callable[[Any], None],
    backup: Callable[[Any], None] | None = None,
) -> dict[str, Any]:
    """CAS, backup, journal and mutation under ONE definition boundary.

    The journal is bookkeeping, never approval authority. A durable executor
    must pass authenticated before/after images again after restart. A journal
    matching neither image is a conflict, not an invitation to overwrite an
    intervening edit. An already-after image finishes bookkeeping without
    replaying mutation or backup. Ordinary exceptions roll back this operation;
    process death leaves the journal for explicit authenticated reconciliation.
    """
    if (
        not transaction_id
        or len(transaction_id) > 128
        or any(
            char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for char in transaction_id
        )
    ):
        raise ValueError("invalid definition transaction identity")
    with definition_lock(root):
        path = Path(root) / "profile-transactions" / (transaction_id + ".json")
        before_token, after_token = commitment(before), commitment(after)
        previous = None
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous.get("before") != before_token or previous.get("after") != after_token:
                raise DefinitionTransactionConflict("transaction identity has different images")
        current = commitment(read())
        if previous is not None and current == after_token:
            result = {**previous, "state": "complete", "reconciled": True}
            _durable_json(path, result)
            return result
        if previous is not None and previous.get("state") == "complete":
            # A later edit may deliberately restore the original bytes. That
            # does not revive a consumed operation's permission to write again.
            raise DefinitionTransactionConflict("completed transaction changed; never replay")
        if current != before_token:
            raise DefinitionTransactionConflict("definition changed; nothing was written")
        record = {"schema": 1, "before": before_token, "after": after_token, "state": "prepared"}
        _durable_json(path, record)
        if backup is not None:
            backup(before)
        try:
            write(after)
            if commitment(read()) != after_token:
                raise DefinitionTransactionConflict("definition write did not match its commitment")
        except Exception:
            # Every supported writer holds the same lock, so no other writer
            # can land between the failed write and its rollback. Failure to
            # restore is visible and leaves a prepared journal for recovery.
            write(before)
            if commitment(read()) != before_token:
                raise DefinitionTransactionConflict(
                    "rollback did not restore its commitment; recovery required"
                )
            record["state"] = "rolled_back"
            _durable_json(path, record)
            raise
        record["state"] = "complete"
        _durable_json(path, record)
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
        active[key] = owner
        path = Path(root) / "profile-transactions" / ("edit-" + uuid.uuid4().hex + ".json")
        record = {
            "schema": 1,
            "state": "prepared",
            "before": commitment(before),
            "before_image": before,
            "after": None,
        }
        try:
            _durable_json(path, record)
            if backup is not None:
                backup(before)
            try:
                yield
            except Exception:
                if commitment(read()) != record["before"]:
                    restore(before)
                    if commitment(read()) != record["before"]:
                        raise DefinitionTransactionConflict(
                            "rollback did not restore its commitment; recovery required"
                        )
                record["state"] = "rolled_back"
                _durable_json(path, record)
                raise
            record.update(state="complete", after=commitment(read()))
            _durable_json(path, record)
        finally:
            active.pop(key, None)


def team_image(registry: Any, team_id: str) -> dict[str, Any] | None:
    """Fresh team metadata AND briefs, under the shared definition boundary."""
    registry._load()
    try:
        return registry.get_team(team_id).model_dump(mode="json")
    except KeyError:
        return None


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
            team_id = str(target.id) if fn.__name__ == "save_team" else str(target)
            from local_operator.teams import Team

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
