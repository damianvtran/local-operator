"""Projection operations must never follow an ancestor into unrelated data."""

import json
import os
from pathlib import Path

import pytest

from local_operator.hub_sync.identity import LinkError
from local_operator.hub_sync.skill_storage import ManagedSkillStorage

VALUE = {"alias": "example", "carried_paths": ["assets/local.bin"]}


def _operate(storage: ManagedSkillStorage, operation: str):
    if operation == "read":
        return storage._projection("example")
    if operation == "write":
        return storage._write_projection("example", VALUE)
    if operation == "path":
        return storage._projection_path("example")
    return storage._retire_projection("example")


def test_retire_rejects_hub_symlink_before_unlink(tmp_path: Path):
    config = tmp_path / "config"
    config.mkdir()
    sessions = tmp_path / "sessions"
    victims = sessions / "skill-projections"
    victims.mkdir(parents=True)
    victim = victims / "example.json"
    victim.write_bytes(b"unrelated session bytes")
    (config / "hub").symlink_to(sessions, target_is_directory=True)
    storage = ManagedSkillStorage(config)

    with pytest.raises(LinkError):
        storage._retire_projection("example")
    assert victim.read_bytes() == b"unrelated session bytes"


@pytest.mark.parametrize("operation", ["read", "write", "retire", "path"])
@pytest.mark.parametrize("component", ["config", "hub", "skill-projections", "leaf"])
def test_symlink_refusal_preserves_victim_and_policy(tmp_path, operation, component):
    config = tmp_path / "config"
    folder = config / "hub" / "skill-projections"
    folder.mkdir(parents=True)
    storage = ManagedSkillStorage(config)
    outside = tmp_path / "sessions"
    victim_folder = outside / "hub" / "skill-projections"
    victim_folder.mkdir(parents=True)
    victim = victim_folder / "example.json"
    victim.write_bytes(b"unrelated session bytes")
    policy = outside / "policy.json"
    policy.write_bytes(b"unrelated link policy bytes")
    if component == "leaf":
        (folder / "example.json").symlink_to(victim)
    else:
        target, replacement = {
            "config": (config, outside),
            "hub": (config / "hub", outside / "hub"),
            "skill-projections": (folder, victim_folder),
        }[component]
        target.rename(tmp_path / "owned-original")
        target.symlink_to(replacement, target_is_directory=True)
    before = {
        path.relative_to(outside): path.read_bytes()
        for path in outside.rglob("*")
        if path.is_file()
    }
    with pytest.raises(LinkError) as error:
        _operate(storage, operation)
    assert error.value.code == "skill-validation"
    assert {
        path.relative_to(outside): path.read_bytes()
        for path in outside.rglob("*")
        if path.is_file()
    } == before
    assert victim.read_bytes() == b"unrelated session bytes"
    assert policy.read_bytes() == b"unrelated link policy bytes"


@pytest.mark.parametrize("operation", ["read", "write", "retire"])
@pytest.mark.parametrize("component", ["hub", "skill-projections"])
def test_ancestor_swap_after_open_keeps_operations_in_owned_directory(
    tmp_path, monkeypatch, operation, component
):
    config = tmp_path / "config"
    folder = config / "hub" / "skill-projections"
    folder.mkdir(parents=True)
    old = {"alias": "example", "carried_paths": []}
    (folder / "example.json").write_text(json.dumps(old))
    storage = ManagedSkillStorage(config)
    outside = tmp_path / "sessions"
    victim_folder = outside / "skill-projections"
    victim_folder.mkdir(parents=True)
    victim = victim_folder / "example.json"
    victim.write_bytes(b"unrelated session bytes")
    policy = outside / "policy.json"
    policy.write_bytes(b"unrelated link policy bytes")
    retained = tmp_path / "owned-original"
    real_open = os.open
    swapped = False

    def swap_after_open(path, flags, *args, **kwargs):
        nonlocal swapped
        descriptor = real_open(path, flags, *args, **kwargs)
        if path == "skill-projections" and not swapped:
            swapped = True
            target = config / "hub" if component == "hub" else folder
            replacement = outside if component == "hub" else victim_folder
            target.rename(retained)
            target.symlink_to(replacement, target_is_directory=True)
        return descriptor

    monkeypatch.setattr(os, "open", swap_after_open)
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {swap_after_open})
    result = _operate(storage, operation)
    assert swapped
    owned_folder = retained / "skill-projections" if component == "hub" else retained
    owned_leaf = owned_folder / "example.json"
    if operation == "read":
        assert result == old
        assert json.loads(owned_leaf.read_text()) == old
    elif operation == "write":
        assert json.loads(owned_leaf.read_text()) == VALUE
    else:
        assert not owned_leaf.exists()
    assert victim.read_bytes() == b"unrelated session bytes"
    assert policy.read_bytes() == b"unrelated link policy bytes"
    assert not list(retained.rglob("*.tmp"))


@pytest.mark.parametrize("operation", ["read", "write", "retire"])
@pytest.mark.parametrize("kind", ["directory", "fifo"])
def test_nonregular_projection_refused_without_blocking(tmp_path, operation, kind):
    storage = ManagedSkillStorage(tmp_path / "config")
    folder = storage.config_dir / "hub" / "skill-projections"
    folder.mkdir(parents=True)
    leaf = folder / "example.json"
    if kind == "directory":
        leaf.mkdir()
    else:
        os.mkfifo(leaf)
    with pytest.raises(LinkError):
        _operate(storage, operation)
    assert leaf.exists()


def test_projection_round_trip_and_missing_retirement(tmp_path):
    storage = ManagedSkillStorage(tmp_path / "config")
    assert storage._projection("example") is None
    storage._retire_projection("example")
    storage._write_projection("example", VALUE)
    assert storage._projection("example") == VALUE
    assert storage._projection_path("example").stat().st_mode & 0o777 == 0o600
    changed = {"alias": "example", "carried_paths": []}
    storage._write_projection("example", changed)
    assert storage._projection("example") == changed
    storage._retire_projection("example")
    assert storage._projection("example") is None
    storage._retire_projection("example")


def test_unavailable_secure_operations_fail_before_touching_projection(tmp_path, monkeypatch):
    storage = ManagedSkillStorage(tmp_path / "config")
    storage._write_projection("example", VALUE)
    leaf = storage._projection_path("example")
    before = leaf.read_bytes()
    monkeypatch.setattr(storage, "_projection_supported", lambda: False)
    for operation in ("read", "write", "retire"):
        with pytest.raises(LinkError) as caught:
            _operate(storage, operation)
        assert caught.value.code == "projection-unsupported"
    assert leaf.read_bytes() == before


@pytest.mark.parametrize("entry", ["preview", "adopt"])
def test_unsupported_adoption_refuses_before_recovery_or_content_writes(
    tmp_path, monkeypatch, entry
):
    from .test_skill_adoption_storage import _setup

    identity, document, baseline, storage, tree, snapshot = _setup(tmp_path)
    before = {path: path.read_bytes() for path in tree.root.rglob("*") if path.is_file()}
    hub_before = {
        path: path.read_bytes()
        for path in (storage.config_dir / "hub").rglob("*")
        if path.is_file()
    }
    events = []
    monkeypatch.setattr(storage, "_projection_supported", lambda: False)
    monkeypatch.setattr(storage, "recover", lambda: events.append("recovery"))
    monkeypatch.setattr(storage, "_finalize", lambda transaction: events.append("finalize"))
    with pytest.raises(LinkError) as error:
        if entry == "preview":
            storage.adoption_subject(identity, document.name)
        else:
            storage.adopt(
                identity,
                document.name,
                document,
                baseline,
                expected_tree_hash=tree.tree_hash,
                snapshot=snapshot,
                checkpoint=events.append,
            )
    assert error.value.code == "projection-unsupported"
    assert not events
    assert {path: path.read_bytes() for path in tree.root.rglob("*") if path.is_file()} == before
    assert not (storage.config_dir / "hub" / "skill-projections").exists()
    assert {
        path: path.read_bytes()
        for path in (storage.config_dir / "hub").rglob("*")
        if path.is_file()
    } == hub_before


class _WinCall:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


class _WindowsProbe:
    """API contract simulator, not evidence of native Windows execution."""

    def __init__(
        self,
        *,
        missing: int | None = 3,
        error=2,
        reparse=None,
        nondirectory=None,
        info_failure=None,
    ):
        import ctypes

        self.ctypes = ctypes
        self.missing = missing
        self.error = error
        self.reparse = reparse
        self.nondirectory = nondirectory
        self.info_failure = info_failure
        self.opened = []
        self.closed = []
        self.live = []
        self.CreateFileW = _WinCall(self.create)
        self.GetFileInformationByHandle = _WinCall(self.info)
        self.GetLastError = _WinCall(lambda: self.error)
        self.CloseHandle = _WinCall(self.close)

    def create(self, path, access, share, security, disposition, flags, template):
        depth = len(self.opened)
        # A deleted/renamed ancestor could redirect absolute child lookup if its
        # handle were closed early or shared DELETE. Check the entire contract.
        assert self.live == list(range(100, 100 + depth))
        assert access == 0x0080
        assert share == 0x0003  # READ|WRITE, critically no DELETE
        assert disposition == 3  # OPEN_EXISTING: probes cannot create anything
        assert flags == 0x02200000  # OPEN_REPARSE_POINT|BACKUP_SEMANTICS
        assert security is None and template is None
        self.opened.append(path)
        if depth == self.missing:
            return self.ctypes.c_void_p(-1).value
        handle = 100 + depth
        self.live.append(handle)
        return handle

    def info(self, handle, pointer):
        depth = handle - 100
        if depth == self.info_failure:
            return 0
        fields = self.ctypes.cast(pointer, self.ctypes.POINTER(self.ctypes.c_uint32))
        fields[0] = 0x0010 if depth != self.nondirectory else 0
        if depth == self.reparse:
            fields[0] |= 0x0400
        return 1

    def close(self, handle):
        assert self.live.pop() == handle
        self.closed.append(handle)
        return 1


@pytest.mark.parametrize("missing", range(4))
@pytest.mark.parametrize("error", [2, 3])
def test_windows_absence_probe_pins_chain_and_closes_reverse(tmp_path, missing, error):
    storage = ManagedSkillStorage(tmp_path / "config")
    kernel = _WindowsProbe(missing=missing, error=error)
    assert storage._projection_windows_absent("example", _kernel=kernel)
    assert not kernel.live
    assert kernel.closed == list(reversed(range(100, 100 + missing)))


@pytest.mark.parametrize("error", [5, 32, 123])
def test_windows_probe_denial_is_not_absence(tmp_path, error):
    storage = ManagedSkillStorage(tmp_path / "config")
    kernel = _WindowsProbe(error=error)
    with pytest.raises(LinkError) as caught:
        storage._projection_windows_absent("example", _kernel=kernel)
    assert caught.value.code == "skill-validation"
    assert not kernel.live


@pytest.mark.parametrize("depth", range(4))
@pytest.mark.parametrize("problem", ["reparse", "info_failure"])
def test_windows_probe_refuses_reparse_and_query_failure(tmp_path, depth, problem):
    storage = ManagedSkillStorage(tmp_path / "config")
    kernel = _WindowsProbe(missing=None, **{problem: depth})
    with pytest.raises(LinkError):
        storage._projection_windows_absent("example", _kernel=kernel)
    assert not kernel.live


@pytest.mark.parametrize("depth", range(3))
def test_windows_probe_refuses_nondirectory_ancestor(tmp_path, depth):
    storage = ManagedSkillStorage(tmp_path / "config")
    kernel = _WindowsProbe(missing=None, nondirectory=depth)
    with pytest.raises(LinkError):
        storage._projection_windows_absent("example", _kernel=kernel)
    assert not kernel.live


def test_windows_existing_leaf_is_never_interpreted_as_absence(tmp_path):
    storage = ManagedSkillStorage(tmp_path / "config")
    kernel = _WindowsProbe(missing=None)
    assert not storage._projection_windows_absent("example", _kernel=kernel)
    assert len(kernel.closed) == 4
    assert not kernel.live


def test_capability_absence_preserves_strict_pull_update_unpull(tmp_path, monkeypatch):
    from dataclasses import replace

    from .test_skill_storage import _input

    identity, document, baseline = _input()
    storage = ManagedSkillStorage(tmp_path / "config")
    monkeypatch.setattr(storage, "_projection_supported", lambda: False)
    storage.apply(identity, document.name, document, baseline, expected_fingerprint=None)
    assert storage.snapshot(identity.local_id).document == document
    changed = document.with_description("Updated routing guidance")
    changed_baseline = replace(baseline, fields=changed.to_json(), fingerprint=changed.fingerprint)
    storage.apply(
        identity,
        document.name,
        changed,
        changed_baseline,
        expected_fingerprint=document.fingerprint,
    )
    assert storage.snapshot(identity.local_id).document == changed
    assert storage.unpull(identity.local_id).content_state == "absent"
    storage.recover_unpull(local_id=identity.local_id)
    assert storage._projection(document.name) is None


@pytest.mark.parametrize(
    "entry", ["apply", "snapshot", "unpull", "recover-unpull", "recover-adoption"]
)
def test_unsupported_existing_projection_refuses_before_any_mutation(tmp_path, monkeypatch, entry):
    from .test_skill_adoption_storage import _setup

    identity, document, baseline, storage, tree, snapshot = _setup(tmp_path)

    def crash(phase):
        if phase == "content-committed":
            raise RuntimeError("crash")

    if entry == "recover-adoption":
        with pytest.raises(RuntimeError, match="crash"):
            storage.adopt(
                identity,
                document.name,
                document,
                baseline,
                expected_tree_hash=tree.tree_hash,
                snapshot=snapshot,
                checkpoint=crash,
            )
    else:
        storage.adopt(
            identity,
            document.name,
            None,
            baseline,
            expected_tree_hash=tree.tree_hash,
            snapshot=snapshot,
            carried_paths=("assets/local.bin",),
        )
        if entry == "recover-unpull":

            def stop(alias):
                raise RuntimeError("crash")

            monkeypatch.setattr(storage, "_retire_projection", stop)
            with pytest.raises(RuntimeError, match="crash"):
                storage.unpull(identity.local_id)

    before = {path: path.read_bytes() for path in storage.config_dir.rglob("*") if path.is_file()}
    monkeypatch.setattr(storage, "_projection_supported", lambda: False)
    with pytest.raises(LinkError) as caught:
        if entry == "apply":
            storage.apply(
                identity,
                document.name,
                document,
                baseline,
                expected_fingerprint=document.fingerprint,
            )
        elif entry == "snapshot":
            storage.snapshot(identity.local_id)
        elif entry == "unpull":
            storage.unpull(identity.local_id)
        elif entry == "recover-unpull":
            storage.recover_unpull(local_id=identity.local_id)
        else:
            storage.recover(local_id=identity.local_id)
    assert caught.value.code == "projection-unsupported"
    assert {path: path.read_bytes() for path in before} == before
    assert {path for path in storage.config_dir.rglob("*") if path.is_file()} == set(before)


def test_windows_probe_close_failure_does_not_report_absence(tmp_path):
    storage = ManagedSkillStorage(tmp_path / "config")
    kernel = _WindowsProbe()
    close = kernel.close

    def failing_close(handle):
        close(handle)
        return 0

    kernel.CloseHandle = _WinCall(failing_close)
    with pytest.raises(LinkError):
        storage._projection_windows_absent("example", _kernel=kernel)
    assert not kernel.live  # Every acquired handle was attempted despite errors.
