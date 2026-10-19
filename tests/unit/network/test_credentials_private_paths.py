"""Root-preparation contract controls; authored source, execution separately gated.

Once installed, the write guard checks tested call-boundary writes against the
ENTIRE disposable fixture envelope, not only config Root. It intentionally permits
old bad ancestor chmod INSIDE that envelope
so a baseline run discriminates the defect, but fails BEFORE Python OS writes
outside it. It is a trusted-test call-boundary guard, not a kernel sandbox for
arbitrary C/native code. SQLite connection paths are checked before its C writer.
Software signing is the existing fixture seam, NOT native presence/install proof.
Test-local env changes do NOT retroactively isolate parent startup/site/imports,
bytecode or cached tempfile paths. Spawned interpreters import before their target
guard. A separately admitted existing controller must establish the startup
profile/source winners and own auxiliaries/threads; these tests do not do that.
"""

from __future__ import annotations

import builtins
import io
import json
import multiprocessing
import os
import sqlite3
import stat
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import unquote, urlsplit

import pytest

from local_operator.network import approvals, identity, store
from local_operator.network.credentials import consent, inheritance, placement
from local_operator.network.types import MemberRecord, MeshRefusal, NetworkRecord
from tests.unit.network.test_credential_inherit import (
    software_signer,
    with_owner_transport,
)
from tests.unit.network.test_credentials_consent import N, P


def _guard_fixture_writes(monkeypatch: pytest.MonkeyPatch, envelope: Path) -> None:
    """Fail before the tested Python writers escape our whole fixture; no chmod masking."""
    envelope = envelope.resolve()
    opened: dict[int, tuple[Path, os.stat_result]] = {}
    original_open, original_close = os.open, os.close
    original_fstat = os.fstat

    def fd_identity(info: os.stat_result) -> tuple[int, int, int]:
        return info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode)

    def checked(value: Any, *, dir_fd: int | None = None) -> Path:
        if isinstance(value, int):
            assert value in opened, "untracked file descriptor mutation refused"
            path, observed = opened[value]
            assert fd_identity(original_fstat(value)) == fd_identity(
                observed
            ), "tracked file descriptor identity changed"
        else:
            path = relative_path(value, dir_fd)
        assert path.resolve().is_relative_to(envelope), "write outside disposable envelope refused"
        return path

    def relative_path(value: Any, dir_fd: int | None) -> Path:
        path = Path(os.fsdecode(value))
        if dir_fd is not None:
            # shutil's safe rmtree uses relative names, not ambient cwd. Only a
            # still-observed directory inside this envelope may supply that base.
            base = checked(dir_fd)
            assert stat.S_ISDIR(original_fstat(dir_fd).st_mode), "dir_fd is not a directory"
            if not path.is_absolute():
                path = base / path
        return path.absolute()

    def ipc_descriptor(fd: int) -> bool:
        # Spawn serializes to a real pipe FD on both Windows and POSIX. Validate
        # the kernel type at the I/O boundary; integers/character devices are not
        # blanket capabilities, and this exception never admits chmod/truncate.
        mode = original_fstat(fd).st_mode
        return stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode)

    def os_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        actual = relative_path(path, kwargs.get("dir_fd"))
        if flags & (os.O_CREAT | os.O_WRONLY | os.O_RDWR | os.O_TRUNC | os.O_APPEND):
            checked(actual)
        fd = original_open(path, flags, *args, **kwargs)
        opened[fd] = actual, original_fstat(fd)
        return fd

    def close(fd: int) -> None:
        try:
            original_close(fd)
        finally:
            opened.pop(fd, None)

    monkeypatch.setattr(os, "open", os_open)
    monkeypatch.setattr(os, "close", close)
    for name, positions in {
        "mkdir": (0,),
        "chmod": (0,),
        "unlink": (0,),
        "remove": (0,),
        "rmdir": (0,),
        "replace": (0, 1),
        "rename": (0, 1),
        "utime": (0,),
        "symlink": (1,),
        "link": (0, 1),
        "truncate": (0,),
    }.items():
        original = getattr(os, name)

        def guarded(
            *args: Any,
            _original: Any = original,
            _positions: tuple[int, ...] = positions,
            **kwargs: Any,
        ) -> Any:
            for position in _positions:
                if len(_positions) == 1:
                    key = "dir_fd"
                else:
                    key = "src_dir_fd" if position == 0 else "dst_dir_fd"
                checked(args[position], dir_fd=kwargs.get(key))
            return _original(*args, **kwargs)

        monkeypatch.setattr(os, name, guarded)
    for name in ("fchmod", "ftruncate", "write"):
        original = getattr(os, name, None)
        if original is None:
            continue

        def guarded_fd(
            fd: int, *args: Any, _original: Any = original, _name: str = name, **kwargs: Any
        ) -> Any:
            if _name != "write" or not ipc_descriptor(fd):
                checked(fd)
            return _original(fd, *args, **kwargs)

        monkeypatch.setattr(os, name, guarded_fd)
    for module in (builtins, io):
        original = module.open

        def file_open(
            file: Any,
            mode: str = "r",
            *args: Any,
            _original: Any = original,
            **kwargs: Any,
        ) -> Any:
            if any(flag in mode for flag in "wax+"):
                if not isinstance(file, int) or not ipc_descriptor(file):
                    checked(file)
            return _original(file, mode, *args, **kwargs)

        monkeypatch.setattr(module, "open", file_open)
    original_connect = sqlite3.connect

    def connect(database: Any, *args: Any, **kwargs: Any) -> Any:
        if database != ":memory:":
            text = os.fsdecode(database)
            if text.startswith("file:"):
                uri = urlsplit(text)
                assert not uri.netloc, "unexpected SQLite host in fixture"
                checked(unquote(uri.path))
            else:
                checked(database)
        return original_connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect)


@pytest.fixture(name="write_envelope")
def write_envelope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Do not borrow real HOME, cached config, temp or inherited child Python paths.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "default-profile"))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    # Fixture-local settings do not undo parent startup/bytecode/tempfile-cache
    # effects. An accepted controller must establish those BEFORE interpreter start.
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("SQLITE_TMPDIR", str(tmp_path))
    _guard_fixture_writes(monkeypatch, tmp_path)
    return tmp_path


@pytest.fixture
def guarded_rig(write_envelope: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Seed valid INPUTS without the old broken prep before public APPLY.

    Same unchanged serializer/private JSON input primitive on baseline and fixed
    source. No PlacementDocument.save, apply, consent lock or chmod is replaced.
    No broker is constructed: this test exercises permission recording, not a
    relay/broker lifecycle. Actual software signatures and engine verification stay.
    """
    from local_operator.operator import trust
    from local_operator.providers import registry
    from local_operator.providers.auth_store import AuthStore
    from tests.unit.network.test_approvals_store import _make_key

    root = write_envelope / "isolated-owner"
    root.mkdir(mode=0o700)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setattr(
        trust,
        "load_anchor",
        lambda uid=None: trust.AnchorLoad(
            anchor=None,
            path=root / "absent-installed-anchor",
            root_owned=False,
            reason="synthetic file-only authority, no host fallback",
            exists=False,
        ),
    )
    _make_key(root)
    own = identity.load_or_mint(root, name="synthetic-owner")
    peer = identity.load_or_mint(write_envelope / "isolated-recipient", name="synthetic-recipient")
    network = NetworkRecord(
        N,
        "synthetic",
        self_device_id=own.device_id,
        created_by=own.device_id,
        members=[
            MemberRecord(
                peer.device_id,
                public_key=peer.to_json()["public_key"],
                role="admin",
                capabilities=["broker_credential"],
            )
        ],
    )
    store.save(network, root)
    definition = registry.ProviderDefinition(
        id=P,
        name="Permission test",
        store_credentials_as=P,
        get_api_key=lambda row: str(row.get("access") or ""),
    )
    monkeypatch.setitem(registry._BY_ID, P, definition)
    auth = AuthStore(db_path=root / "auth.db")
    try:
        auth.upsert_credential(
            P,
            {
                "type": "oauth",
                "email": "synthetic@example.invalid",
                "access": "synthetic-private-marker",
                "refresh": "synthetic-refresh-marker",
                "expires": int((time.time() + 3600) * 1000),
                "account_id": "account-one",
                "org_id": "org-one",
            },
        )
        doc = placement.PlacementDocument(N, root=root, written_by=own.device_id)
        doc.declare(
            P,
            owner_device=own.device_id,
            owner_device_name="synthetic-owner",
            provider=P,
            kind="oauth",
            by=own.device_id,
        )
        doc.grant(P, peer.device_id, by=own.device_id)
        payload = doc.to_json()
        target = placement.placement_path(N, root)
        directory = root
        for component in target.parent.relative_to(root).parts:
            directory = directory / component
            directory.mkdir(mode=0o700, exist_ok=True)
        # Unchanged input writer, NOT the defective preparation or tested engine.
        store._write_private_json(target, payload)
        # The serializer deliberately stamps a fresh written_at on every call.
        # Check the exact disk snapshot and all authoritative loader fields;
        # a second serialization is not an equality oracle for that clock.
        assert json.loads(target.read_text()) == payload
        loaded = placement.PlacementDocument.load(N, root)
        assert (loaded.network_id, loaded.epoch, loaded.written_by, loaded.entries) == (
            doc.network_id,
            doc.epoch,
            doc.written_by,
            doc.entries,
        )
        yield SimpleNamespace(root=root, own=own, peer=peer, auth=auth)
    finally:
        auth.close()


def test_write_guard_accepts_only_validated_ipc_not_regular_unknown_fds(
    write_envelope: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep spawn's pipe writer distinct from disk ownership, without a guard gap."""
    envelope = write_envelope / "owned"
    envelope.mkdir()
    inside, outside = envelope / "inside", write_envelope / "outside"
    inside.write_bytes(b"inside-sentinel")
    outside.write_bytes(b"outside-sentinel")
    outer_guards = (os.open, os.unlink, os.rmdir, os.write, io.open, builtins.open)
    raw_open = os.open
    unknown = raw_open(inside, os.O_RDWR)
    foreign = raw_open(outside, os.O_RDWR)
    read_fd, write_fd = os.pipe()
    try:
        # The fixture's whole-envelope guard must outlive this deliberately
        # narrower negative cell, including inherited fixture cleanup.
        with monkeypatch.context() as inner:
            _guard_fixture_writes(inner, envelope)
            with io.open(write_fd, "wb", closefd=False) as pipe:
                pipe.write(b"a")
                pipe.flush()
            assert os.read(read_fd, 1) == b"a"
            assert os.write(write_fd, b"b") == 1
            assert os.read(read_fd, 1) == b"b"
            for fd in (unknown, foreign):
                with pytest.raises(AssertionError, match="untracked"):
                    io.open(fd, "wb", closefd=False)
                with pytest.raises(AssertionError, match="untracked"):
                    os.write(fd, b"not-written")
            with pytest.raises(AssertionError, match="untracked"):
                os.ftruncate(write_fd, 0)
            # Even an observed read-only disk open outside the envelope is NOT a
            # later write capability; no broad integer-FD or non-regular fallback.
            tracked_foreign = os.open(outside, os.O_RDONLY)
            try:
                with pytest.raises(AssertionError, match="outside disposable"):
                    os.write(tracked_foreign, b"not-written")
            finally:
                os.close(tracked_foreign)
            assert inside.read_bytes() == b"inside-sentinel"
            assert outside.read_bytes() == b"outside-sentinel"
    finally:
        for fd in (unknown, foreign, read_fd, write_fd):
            os.close(fd)
        assert (os.open, os.unlink, os.rmdir, os.write, io.open, builtins.open) == outer_guards


def test_write_guard_qualifies_owned_relative_cleanup_without_outside_writes(
    write_envelope: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise real rmtree plus retained path/dir_fd refusal, not guard removal."""
    import shutil

    envelope = write_envelope / "owned"
    envelope.mkdir()
    nested = envelope / "nested"
    nested.mkdir()
    (nested / "sentinel").write_bytes(b"owned")
    outside = write_envelope / "outside"
    outside.mkdir()
    (outside / "sentinel").write_bytes(b"foreign")
    outer_guards = (os.open, os.unlink, os.rmdir, os.write, io.open, builtins.open)
    # Both guard layers wrap os functions; native capability registry entries
    # retain their original names, so wrapper identity must not skip dir_fd cases.
    supports_relative = {"rmdir", "open"} <= {fn.__name__ for fn in os.supports_dir_fd}
    with monkeypatch.context() as inner:
        _guard_fixture_writes(inner, envelope)
        shutil.rmtree(nested)
        assert not nested.exists()
        with pytest.raises(AssertionError, match="outside disposable"):
            os.unlink(outside / "sentinel")
        if supports_relative:
            directory = os.open(outside, os.O_RDONLY)
            owned_directory = os.open(envelope, os.O_RDONLY)
            try:
                with pytest.raises(AssertionError, match="outside disposable"):
                    os.unlink("sentinel", dir_fd=directory)
                with pytest.raises(AssertionError, match="outside disposable"):
                    os.rmdir("../outside", dir_fd=owned_directory)
            finally:
                os.close(directory)
                os.close(owned_directory)
        assert (outside / "sentinel").read_bytes() == b"foreign"
    assert (os.open, os.unlink, os.rmdir, os.write, io.open, builtins.open) == outer_guards


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_public_writer_preserves_existing_root_and_ancestors(write_envelope: Path) -> None:
    parent = write_envelope / "owned-parent"
    parent.mkdir(mode=0o755)
    parent.chmod(0o755)
    root = parent / "custom profile with spaces"
    root.mkdir(mode=0o750)
    root.chmod(0o750)
    before = (_mode(parent), _mode(root), _mode(write_envelope))
    target = placement.credentials_dir_for("n_fixture", root)
    assert before == (_mode(parent), _mode(root), _mode(write_envelope))
    if os.name != "nt":
        private_paths = (root / "network", root / "network/credentials", target)
        assert all(_mode(p) == 0o700 for p in private_paths)


def test_missing_root_creates_only_leaf_and_missing_parent_refuses(write_envelope: Path) -> None:
    root = write_envelope / "new-profile"
    placement.credentials_dir_for("n_fixture", root)
    if os.name != "nt":
        assert _mode(root) == 0o700
    with pytest.raises(MeshRefusal) as refused:
        placement.credentials_dir_for("n_fixture", write_envelope / "missing-parent/profile")
    assert refused.value.code == "profile_parent_missing"
    assert not (write_envelope / "missing-parent").exists()


def test_explicit_synthetic_home_is_a_valid_unchanged_root(write_envelope: Path) -> None:
    write_envelope.chmod(0o755)
    before = _mode(write_envelope)
    placement.credentials_dir_for("n_fixture", write_envelope)
    assert _mode(write_envelope) == before


def test_full_inheritance_records_permission_without_ancestor_mode_changes(
    guarded_rig: Any, write_envelope: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = guarded_rig
    with_owner_transport(rig)
    software_signer(rig.root, monkeypatch)
    # These deliberately NON-private sentinels expose the original defect. Never
    # pre-chmod ancestors to0700, suppress chmod, or replace the actual apply engine.
    write_envelope.chmod(0o755)
    rig.root.chmod(0o750)
    sentinel = write_envelope / "ancestor-sentinel"
    sentinel.write_bytes(b"fixture-parent-byte-sentinel")
    before = (_mode(write_envelope), _mode(rig.root), sentinel.read_bytes())
    record = inheritance.request(
        N,
        rig.own.device_id,
        rig.peer.device_id,
        P,
        account="account-one",
        org="org-one",
        root=rig.root,
    )
    decided_at = time.time()
    signature = approvals.sign_decision(
        kind=record["kind"],
        request_id=record["request_id"],
        request_digest=record["request_digest"],
        decided_at=decided_at,
    )
    approvals.approve(
        record["approval_id"], signature_hex=signature, decided_at=decided_at, root=rig.root
    )
    # The setup must already have finished: this receipt distinguishes a real
    # public APPLY/begin_run failure from a fixture/preparation/import failure.
    assert approvals.load_record(record["approval_id"], root=rig.root)["state"] == "approved"
    phase = {
        "stage": "calling_public_inheritance_apply",
        "apply_call_attempts": 1,
        "approval_state_before": "approved",
        "returned_permission_recorded": False,
        "persisted_run_state_seen": False,
        "parent_mode_before": before[0],
        "root_mode_before": before[1],
    }
    result = None
    try:
        result = inheritance.apply(
            record["approval_id"],
            {
                "owner_device": rig.own.device_id,
                "network_id": N,
                "recipient_device": rig.peer.device_id,
                "provider": P,
                "account": "account-one",
                "org": "org-one",
            },
            rig.root,
        )
    finally:
        observed = approvals.load_record(record["approval_id"], root=rig.root)
        state = observed["state"]
        phase["persisted_run_state_seen"] = state in ("connecting", "failed", "connected")
        phase["approval_state_after"] = (
            state if state in ("approved", "connecting", "failed", "connected") else "other"
        )
        phase["stage"] = "public_apply_returned" if result is not None else "public_apply_raised"
        phase["returned_permission_recorded"] = bool(
            result and result.get("kind") == "permission_recorded"
        )
        phase["parent_mode_after"] = _mode(write_envelope)
        phase["root_mode_after"] = _mode(rig.root)
        # Bounded structural fields only. The original exception still propagates:
        # a baseline guard refusal is NOT converted into a successful test.
        (write_envelope / "public-apply-phase.json").write_text(json.dumps(phase), encoding="utf-8")
    assert phase["apply_call_attempts"] == 1 and phase["persisted_run_state_seen"] is True
    assert result is not None and result["kind"] == "permission_recorded"
    assert approvals.load_record(record["approval_id"], root=rig.root)["state"] == "connected"
    policy = consent.ConsentDocument(N, rig.own.device_id, rig.root).load()
    assert policy["recipients"][rig.peer.device_id]["approval"]["ref"] == record["approval_id"]
    assert before == (_mode(write_envelope), _mode(rig.root), sentinel.read_bytes())
    private = placement.placement_path(N, rig.root).parent
    if os.name != "nt":
        assert all(_mode(p) == 0o700 for p in (rig.root / "network", private.parent, private))
        assert _mode(private / "consent.json") == 0o600
        assert _mode(private / ".consent.lock") == 0o600


@pytest.mark.parametrize("bad", ["symlink", "file", "escape"])
def test_descendant_redirect_or_type_is_refused_before_prep(write_envelope: Path, bad: str) -> None:
    root = write_envelope / "profile"
    root.mkdir()
    target = root / "network"
    if bad == "symlink":
        outside = write_envelope / "different-profile"
        outside.mkdir()
        target.symlink_to(outside, target_is_directory=True)
    elif bad == "file":
        target.write_bytes(b"not-a-directory")
    else:
        target = root / "../escape"
    with pytest.raises(MeshRefusal):
        placement._ensure_private_dir(target, root=root.resolve())


def test_lock_symlink_or_nonregular_never_reaches_try_lock(
    write_envelope: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.wakes import lock

    root = write_envelope / "profile"
    parent = placement.credentials_dir_for("n_fixture", root)
    target = parent / ".operational.lock"
    target.symlink_to(parent / "absent-leaf")
    calls: list[bool] = []
    monkeypatch.setattr(lock, "_try_lock", lambda fd: calls.append(True))
    with pytest.raises(MeshRefusal):
        with placement._cross_process_lock(target, root=root.resolve()):
            raise AssertionError("redirected lock was admitted")
    assert calls == []


@pytest.mark.parametrize("flavor", ["nt", "posix-no-flags"])
def test_checked_platform_fallback_is_not_a_blanket_refusal(
    write_envelope: Path, monkeypatch: pytest.MonkeyPatch, flavor: str
) -> None:
    # Dispatch simulation using REAL in-envelope filesystem operations; not native
    # Windows locking, drive/UNC availability, or ACL confidentiality evidence.
    fake = SimpleNamespace(
        **{
            name: getattr(os, name)
            for name in dir(os)
            if name not in ("O_NOFOLLOW", "O_DIRECTORY", "fchmod")
        }
    )
    fake.name = "nt" if flavor == "nt" else "posix"
    monkeypatch.setattr(placement, "os", fake)
    root = write_envelope / "profile"
    target = placement.credentials_dir_for("n_fixture", root)
    assert target.is_dir()
    fd = placement._open_lock(target / ".test.lock")
    os.close(fd)


def test_observed_windows_reparse_is_refused_as_data_only(
    write_envelope: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = write_envelope.stat()

    def attributes(value: object, name: str, default: Any = None) -> Any:
        # Keep a real stat_result input. Only its Windows-only attribute READ is
        # modeled on POSIX; this is predicate data coverage, not native OS proof.
        if value is info and name == "st_file_attributes":
            return 0x400
        return getattr(value, name, default)

    monkeypatch.setattr(placement, "getattr", attributes, raising=False)
    with pytest.raises(MeshRefusal):
        placement._checked_info(info, directory=True)


def _lock_process(envelope: str, root: str, target: str, connection: Any, holder: bool) -> None:
    with pytest.MonkeyPatch.context() as patch:
        _guard_fixture_writes(patch, Path(envelope))
        path = Path(target)
        if holder:
            with placement._cross_process_lock(path, root=Path(root)):
                info = path.stat()
                connection.send(("held", info.st_dev, info.st_ino))
                assert connection.poll(10), "owner release command missing"
                assert connection.recv() == "release"
            connection.send(("released",))
        else:
            # Exercise the existing actual busy/timeout primitive without a sleep
            # race: the parent waits for the other process's HELD receipt first.
            placement.PLACEMENT_LOCK_WAIT_S = 0
            assert connection.poll(10)
            assert connection.recv() == "busy"
            try:
                with placement._cross_process_lock(path, root=Path(root)):
                    connection.send(("unexpected-owner",))
            except MeshRefusal as error:
                connection.send((error.code, path.stat().st_dev, path.stat().st_ino))
            assert connection.poll(10)
            assert connection.recv() == "free"
            with placement._cross_process_lock(path, root=Path(root)):
                connection.send(("acquired", path.stat().st_dev, path.stat().st_ino))


def test_two_process_lock_retains_inode_and_actual_timeout(write_envelope: Path) -> None:
    root = (write_envelope / "profile").resolve()
    target = placement.credentials_dir_for("n_fixture", root) / ".operational.lock"
    context = multiprocessing.get_context("spawn")
    left, child_left = context.Pipe()
    right, child_right = context.Pipe()
    holder = context.Process(
        target=_lock_process,
        args=(str(write_envelope), str(root), str(target), child_left, True),
    )
    contender = context.Process(
        target=_lock_process,
        args=(str(write_envelope), str(root), str(target), child_right, False),
    )
    processes = [holder, contender]
    try:
        holder.start()
        assert left.poll(15)
        first = left.recv()
        assert first[0] == "held"
        contender.start()
        right.send("busy")
        assert right.poll(15)
        busy = right.recv()
        assert busy[0] == "busy" and busy[1:] == first[1:]
        left.send("release")
        assert left.poll(15) and left.recv() == ("released",)
        right.send("free")
        assert right.poll(15)
        acquired = right.recv()
        assert acquired[0] == "acquired" and acquired[1:] == first[1:]
        for process in processes:
            process.join(15)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.pid is not None and process.is_alive():
                process.terminate()
                process.join(5)
                if process.is_alive():
                    process.kill()
                    process.join(5)
        left.close()
        right.close()
        child_left.close()
        child_right.close()


def test_document_lock_retains_thread_then_process_order(
    write_envelope: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = (write_envelope / "profile").resolve()
    path = placement.placement_path("n_fixture", root)
    events: list[str] = []
    original_thread, original_process = store._write_lock, placement._cross_process_lock

    @contextmanager
    def thread(target: Path):
        with original_thread(target):
            events.append("thread")
            yield

    @contextmanager
    def process(target: Path, *, root: Path):
        with original_process(target, root=root):
            events.append("process")
            yield

    monkeypatch.setattr(store, "_write_lock", thread)
    monkeypatch.setattr(placement, "_cross_process_lock", process)
    with placement._document_lock(path, root=root):
        events.append("body")
    assert events == ["thread", "process", "body"]


def test_observed_lock_identity_change_refuses_before_primitive(
    write_envelope: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.wakes import lock

    root = (write_envelope / "profile").resolve()
    target = placement.credentials_dir_for("n_fixture", root) / ".identity.lock"
    target.write_bytes(b"")
    real_fstat = os.fstat
    leaf = target.stat()
    attempts: list[bool] = []

    def changed(fd: int) -> Any:
        observed = real_fstat(fd)
        if (observed.st_dev, observed.st_ino) != (leaf.st_dev, leaf.st_ino):
            return observed
        fields = list(observed)
        fields[1] += 1  # st_ino: model only the observed identity change.
        return os.stat_result(fields)

    # Boundary observation only, not a fake successful lock. The actual lock
    # leaf open/identity predicate runs and the primitive must never be reached.
    monkeypatch.setattr(placement.os, "fstat", changed)
    monkeypatch.setattr(lock, "_try_lock", lambda fd: attempts.append(True))
    with pytest.raises(MeshRefusal):
        with placement._cross_process_lock(target, root=root):
            pytest.fail("identity mismatch admitted")
    assert attempts == []


def test_private_descendant_owner_mismatch_predicate_is_fail_closed(write_envelope: Path) -> None:
    if os.name == "nt" or not hasattr(os, "geteuid"):
        pytest.skip("UID ownership is not a meaningful native predicate on this platform")
    fields = list(write_envelope.stat())
    fields[4] = os.geteuid() + 1  # st_uid, preserving the actual stat_result shape.
    observed = os.stat_result(fields)
    with pytest.raises(MeshRefusal):
        placement._checked_info(observed, directory=True)


def test_all_stateful_document_roots_survive_changed_ambient_config(
    write_envelope: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.network.credentials.finite_journal import Journal
    from local_operator.network.credentials.lifecycle import LifecycleStore
    from local_operator.network.credentials.state import PlacementState
    from local_operator.network.credentials.sync import SyncState

    chosen, later = write_envelope / "chosen", write_envelope / "later"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(chosen))
    projection = placement.PlacementDocument("n_fixture")
    objects = [
        projection,
        PlacementState("n_fixture"),
        SyncState("n_fixture"),
        consent.ConsentDocument("n_fixture", "d_owner"),
        LifecycleStore("n_fixture"),
        Journal("n_fixture", "d_owner"),
    ]
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(later))
    for document in objects:
        assert document.root == chosen.resolve()
        assert document.path.is_relative_to(chosen.resolve())
    assert not chosen.exists() and not later.exists()
    # Real writer, not a made-up success object: construction did not initialize
    # anything, and later save still chooses the original root.
    projection.save()
    assert projection.path.exists() and not later.exists()


def test_sync_mutate_binds_import_and_one_root_through_load_save(
    write_envelope: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.network.credentials import sync

    first, second = write_envelope / "sync-first", write_envelope / "sync-second"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(first))
    with sync.mutate("n_fixture") as document:
        assert document.root == first.resolve()
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(second))
        document.record_generation(P, 1, "a" * 64, 1)
    readback = sync.SyncState.load("n_fixture", root=first)
    row = readback.generation(P)
    assert row is not None and row["gen"] == 1
    assert readback.path.exists() and not second.exists()
