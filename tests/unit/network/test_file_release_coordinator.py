"""Isolated durable mechanics and raw TCP, NOT actual PeerLink/source closure.

Every enabled instance (including fresh children) uses an injected OS-home AND
anchor lookup before initialization. HOME/XDG alone cannot isolate this module.
No real native anchor, identity store, installer, or production publisher runs.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import hashlib
import json
import os
import select
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest

from local_operator.network import file_release_coordinator as module
from local_operator.network.file_release_coordinator import (
    CoordinatorIndeterminate,
    CoordinatorObservation,
    CoordinatorReceipt,
    CoordinatorRefusal,
    EffectIdentity,
    FileReleaseCoordinator,
    MutationOutcome,
    MutationReceipt,
    VerifiedOutcome,
)

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux-only coordinator mechanics")


@pytest.fixture
def owner(tmp_path, monkeypatch):
    home = tmp_path / "os-owner"
    home.mkdir(mode=0o700)
    anchor = tmp_path / "uninstalled-system-anchor" / "owner.json"
    monkeypatch.setattr(module, "_owner_home", lambda uid: home)
    monkeypatch.setattr(module, "_anchor_path", lambda uid: anchor)
    return home, anchor


def effect():
    return EffectIdentity("toy-publish", "profile-A:toy-target")


def enabled():
    return FileReleaseCoordinator(file_release_coordinator_enabled=True)


def domain(c):
    _, home, suffix = c._location()
    return home.joinpath(*suffix)


def initialized(owner):
    c = enabled()
    c.initialize()
    return c


# Fresh interpreters, not fork-inherited Python locks/resolvers. This code is a
# rig-only adapter and injects BOTH location lookups before enabling anything.
_CHILD = """
import hashlib, json, os, sys
from pathlib import Path
from local_operator.network import file_release_coordinator as m
home, anchor, action = sys.argv[1:4]
m._owner_home = lambda uid: Path(home)
m._anchor_path = lambda uid: Path(anchor)
c = m.FileReleaseCoordinator(file_release_coordinator_enabled=True)
effect = lambda: m.EffectIdentity('toy-publish', 'profile-A:toy-target')
if action == 'capture':
    try:
        print('CAPTURE', c.capture().revision)
    except m.CoordinatorRefusal:
        print('DENIED')
elif action == 'guard-old':
    receipt = m.CoordinatorReceipt(**json.loads(sys.argv[4]))
    try:
        with c.guard(receipt):
            print('POSITIVE')
    except m.CoordinatorRefusal:
        print('DENIED')
elif action == 'mutate':
    try:
        tx = c.begin_mutation(effect=effect())
        target = Path(home) / 'toy-target'
        staged = Path(home) / 'toy-stage'
        with staged.open('wb') as f:
            f.write(b'verified child publication')
            f.flush()
            os.fsync(f.fileno())
        os.replace(staged, target)
        directory = os.open(home, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        observed = target.read_bytes()
        assert observed == b'verified child publication'
        verified = m.VerifiedOutcome(tx.effect, hashlib.sha256(observed).hexdigest())
        c.finish_mutation(tx, verified_outcome=verified)
        print('COMMITTED')
    except m.CoordinatorRefusal:
        print('DENIED')
elif action in ('crash-pending', 'crash-before-toy-rename', 'crash-published'):
    tx = c.begin_mutation(effect=effect())
    if action != 'crash-pending':
        target = Path(home) / 'toy-target'
        staged = Path(home) / 'toy-stage'
        with staged.open('wb') as f:
            f.write(b'independently verified toy publication')
            f.flush()
            os.fsync(f.fileno())
        if action == 'crash-before-toy-rename':
            os._exit(23)
        os.replace(staged, target)
        directory = os.open(home, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        assert target.read_bytes() == b'independently verified toy publication'
    os._exit(23)
"""


def child(owner, action, *extra):
    home, anchor = owner
    env = dict(os.environ, PYTHONPATH=str(Path(module.__file__).parents[2]))
    return subprocess.run(
        [sys.executable, "-c", _CHILD, str(home), str(anchor), action, *extra],
        cwd=home,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,  # test-harness hang detection, not a coordinator policy
        check=False,
    )


def assert_child(owner, action, expected, rc=0):
    result = child(owner, action)
    assert (result.returncode, result.stdout.strip(), result.stderr) == (rc, expected, "")


def assert_old_denied_in_fresh_process(owner, receipt):
    result = child(owner, "guard-old", json.dumps(dataclasses.asdict(receipt)))
    assert (result.returncode, result.stdout.strip(), result.stderr) == (0, "DENIED", "")


def state(c):
    # Inspect through the application-owned strict loader, not a fake success
    # file. Corruption tests below deliberately damage its real writer output.
    with c._locked() as (fd, domain_key, node):
        return c._load(fd, node, domain_key)


def damage(c, name, transform):
    # Negative controls only. Positive fixtures always use initialize/mutations.
    path = domain(c) / name
    payload = json.loads(path.read_bytes())
    path.write_bytes(module._encode(transform(payload)))


def test_default_off_refuses_without_location_or_file_effects(tmp_path, monkeypatch):
    def forbidden(*args):
        pytest.fail("disabled coordinator resolved an OS location")

    monkeypatch.setattr(module, "_owner_home", forbidden)
    monkeypatch.setattr(module, "_anchor_path", forbidden)
    c = FileReleaseCoordinator()
    for operation in (c.initialize, c.capture, lambda: c.begin_mutation(effect=effect())):
        with pytest.raises(CoordinatorRefusal):
            operation()
    receipt = CoordinatorReceipt("unused", str(uuid4()), 0)
    with pytest.raises(CoordinatorRefusal), c.guard(receipt):
        pytest.fail("disabled guard became positive")
    with pytest.raises(CoordinatorRefusal):
        c.finish_mutation(
            MutationReceipt("unused", str(uuid4()), str(uuid4()), effect()),
            verified_outcome=VerifiedOutcome(
                effect(), hashlib.sha256(b"negative-only").hexdigest()
            ),
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("flag", [0, 1, None, "true", object()])
def test_enable_requires_exact_bool(flag):
    with pytest.raises(TypeError):
        FileReleaseCoordinator(file_release_coordinator_enabled=flag)


@pytest.mark.parametrize("platform", ["darwin", "win32", "unknown"])
def test_unsupported_platform_refuses_before_lookup_or_effects(platform, tmp_path, monkeypatch):
    monkeypatch.setattr(module.sys, "platform", platform)
    monkeypatch.setattr(module, "_owner_home", lambda uid: pytest.fail("unsupported lookup"))
    with pytest.raises(CoordinatorRefusal):
        enabled().initialize()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kind", ["relative", "root", "missing-account", "relative-anchor"])
def test_unresolved_owner_never_falls_back_to_profile(kind, owner, tmp_path, monkeypatch):
    if kind == "missing-account":

        def missing(uid):
            raise KeyError(uid)

        monkeypatch.setattr(module, "_owner_home", missing)
    elif kind == "relative-anchor":
        monkeypatch.setattr(module, "_anchor_path", lambda uid: Path("relative"))
    else:
        monkeypatch.setattr(
            module, "_owner_home", lambda uid: Path("relative" if kind == "relative" else "/")
        )
    monkeypatch.setenv("HOME", str(tmp_path / "unrelated-profile"))
    with pytest.raises(CoordinatorRefusal):
        enabled().initialize()
    assert list(owner[0].iterdir()) == []


def test_profiles_share_owner_anchor_domain_and_key_rotation(owner, tmp_path, monkeypatch):
    c = initialized(owner)
    original = domain(c)
    receipt = c.capture()
    for profile in ("first", "second"):
        for key in ("HOME", "XDG_STATE_HOME", "XDG_CONFIG_HOME", "LOCAL_OPERATOR_CONFIG_DIR"):
            monkeypatch.setenv(key, str(tmp_path / profile / key))
        other = enabled()
        assert domain(other) == original
        assert other.capture() == receipt
    # This rig-only anchor lives under OUR tmp_path, not the native anchor root.
    owner[1].parent.mkdir(mode=0o700)
    owner[1].write_bytes(b"first synthetic key")
    owner[1].write_bytes(b"rotated synthetic key")
    assert domain(enabled()) == original
    assert enabled().capture() == receipt


def test_real_private_files_strict_loader_roundtrip_and_fresh_process(owner):
    c = initialized(owner)
    receipt = c.capture()
    assert receipt.revision == 0
    with c.guard(receipt) as observation:
        assert observation == CoordinatorObservation(True)
    root = domain(c)
    assert {p.name for p in root.iterdir()} == {".mutation.lock", "initialized.json", "state.json"}
    assert root.stat().st_mode & 0o777 == 0o700
    for path in root.iterdir():
        info = path.stat()
        assert info.st_mode & 0o777 == 0o600
        assert info.st_uid == os.getuid() and info.st_nlink == 1
    assert state(c).pending == ()
    assert_child(owner, "capture", "CAPTURE 0")
    assert_child(owner, "mutate", "COMMITTED")
    assert enabled().capture().revision == 2
    with pytest.raises(CoordinatorRefusal), c.guard(receipt):
        pytest.fail("old receipt survived a completed mutation")


def test_no_read_bootstrap_or_reinitialization_after_missing_state(owner):
    c = enabled()
    with pytest.raises(CoordinatorRefusal):
        c.capture()
    assert not list(owner[0].iterdir())
    c.initialize()
    (domain(c) / "state.json").unlink()
    with pytest.raises(CoordinatorRefusal):
        enabled().capture()
    with pytest.raises(CoordinatorRefusal):
        enabled().initialize()
    assert not (domain(c) / "state.json").exists()


def test_overlapping_transactions_exact_ownership_and_verified_toy_publication(owner):
    c = initialized(owner)
    old = c.capture()
    first = c.begin_mutation(effect=effect())
    second = enabled().begin_mutation(effect=effect())
    assert first.transaction != second.transaction
    assert state(c).revision == 2
    assert set(state(c).pending) == {first, second}
    with pytest.raises(CoordinatorRefusal):
        enabled().capture()
    # External publication + independent verification OUTSIDE coordinator lock.
    target = owner[0] / "toy-target"
    with target.open("wb") as handle:
        handle.write(b"real intended bytes")
        handle.flush()
        os.fsync(handle.fileno())
    observed = target.read_bytes()
    assert observed == b"real intended bytes"
    verified = VerifiedOutcome(first.effect, hashlib.sha256(observed).hexdigest())
    result = c.finish_mutation(first, verified_outcome=verified)
    assert result.outcome == MutationOutcome.FINISHED
    assert state(c).pending == (second,)
    with pytest.raises(CoordinatorRefusal):
        enabled().capture()
    assert target.read_bytes() == observed
    c.finish_mutation(
        second,
        verified_outcome=VerifiedOutcome(
            second.effect, hashlib.sha256(target.read_bytes()).hexdigest()
        ),
    )
    assert c.capture().revision == 4
    assert state(c).pending == ()
    with pytest.raises(CoordinatorRefusal), c.guard(old):
        pytest.fail("pre-begin receipt was resurrected")
    with pytest.raises(CoordinatorRefusal):
        enabled().finish_mutation(first, verified_outcome=verified)


@pytest.mark.parametrize("outcome", [False, True, None, 1, "yes", object(), lambda: True])
def test_failed_unknown_or_truthy_external_outcome_never_clears_pending(owner, outcome):
    c = initialized(owner)
    tx = c.begin_mutation(effect=effect())
    with pytest.raises(CoordinatorRefusal):
        c.finish_mutation(tx, verified_outcome=outcome)
    assert state(enabled()).pending == (tx,)
    assert_child(owner, "capture", "DENIED")


@pytest.mark.parametrize("action", ["crash-pending", "crash-before-toy-rename", "crash-published"])
def test_pending_survives_actual_process_crash_and_restart(owner, action):
    c = initialized(owner)
    old = c.capture()
    assert_child(owner, action, "", rc=23)
    fresh = enabled()
    assert len(state(fresh).pending) == 1
    assert state(fresh).revision == 1
    assert_child(owner, "capture", "DENIED")
    assert_old_denied_in_fresh_process(owner, old)
    with pytest.raises(CoordinatorRefusal), fresh.guard(old):
        pytest.fail("crash resurrected old receipt")
    if action == "crash-published":
        assert (owner[0] / "toy-target").read_bytes() == b"independently verified toy publication"
    else:
        assert not (owner[0] / "toy-target").exists()
        if action == "crash-before-toy-rename":
            assert (
                owner[0] / "toy-stage"
            ).read_bytes() == b"independently verified toy publication"


def test_stable_lock_cross_process_and_thread_contention(owner):
    c = initialized(owner)
    node = (domain(c) / ".mutation.lock").stat().st_ino
    with c._locked():
        assert_child(owner, "mutate", "DENIED")
        with pytest.raises(CoordinatorRefusal):
            enabled().capture()
    assert_child(owner, "mutate", "COMMITTED")
    assert (domain(c) / ".mutation.lock").stat().st_ino == node


def publish_and_verify(owner, content=b"actual intended mutation"):
    target = owner[0] / "toy-target"
    staged = owner[0] / "toy-stage"
    with staged.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staged, target)
    directory = os.open(owner[0], os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    observed = target.read_bytes()
    assert observed == content
    return VerifiedOutcome(effect(), hashlib.sha256(observed).hexdigest())


@pytest.mark.parametrize("operation", ["begin", "finish"])
@pytest.mark.parametrize(
    "fault",
    [
        "file-fsync",
        "before-rename",
        "after-rename",
        "parent-fsync",
        "readback-exception",
        "readback-mismatch",
        "final-path-check",
    ],
)
def test_commit_ack_fault_matrix_old_receipt_never_resurrects(owner, monkeypatch, operation, fault):
    c = initialized(owner)
    old = c.capture()
    tx = None
    verified = False
    if operation == "finish":
        tx = c.begin_mutation(effect=effect())
        verified = publish_and_verify(owner)
        assert type(verified) is VerifiedOutcome
    original_fsync = os.fsync
    original_replace = os.replace
    original_read = c._read
    original_check = c._check_directory
    published = False

    def fsync(fd):
        path = Path(os.readlink(f"/proc/self/fd/{fd}"))
        if fault == "file-fsync" and path.name.startswith(".state.json."):
            raise OSError("injected state file fsync failure")
        if fault == "parent-fsync" and published and path == domain(c):
            raise OSError("injected post-rename parent fsync failure")
        return original_fsync(fd)

    def replace(src, dst, **kwargs):
        nonlocal published
        if dst == "state.json" and fault == "before-rename":
            raise OSError("injected before visibility linearization")
        result = original_replace(src, dst, **kwargs)
        if dst == "state.json":
            published = True
            if fault == "after-rename":
                raise OSError("injected after visibility linearization")
        return result

    def read(fd, name):
        if published and name == "state.json" and fault == "readback-exception":
            raise OSError("injected final readback failure")
        actual = original_read(fd, name)
        if published and name == "state.json" and fault == "readback-mismatch":
            revision = actual["revision"]
            assert type(revision) is int
            return dict(actual, revision=revision + 1)
        return actual

    def check(nodes):
        if published and fault == "final-path-check":
            raise CoordinatorRefusal("injected final identity check failure")
        return original_check(nodes)

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", fsync)
        patch.setattr(os, "replace", replace)
        patch.setattr(c, "_read", read)
        patch.setattr(c, "_check_directory", check)
        with pytest.raises(CoordinatorIndeterminate):
            if operation == "begin":
                c.begin_mutation(effect=effect())
            else:
                assert type(tx) is MutationReceipt and type(verified) is VerifiedOutcome
                c.finish_mutation(tx, verified_outcome=verified)
    # Failed caller never gets an acknowledgement and its handle latches denial,
    # even if the final verified completion was already visible on disk.
    with pytest.raises(CoordinatorRefusal):
        c.capture()
    fresh = enabled()
    disk = state(fresh)
    if operation == "begin" and not published:
        # No acknowledged begin AND no external effect: there is no impossible
        # promise to poison an unchanged old committed state across restart.
        assert disk.revision == 0 and disk.pending == ()
        assert_child(owner, "capture", "CAPTURE 0")
    else:
        assert disk.revision == (2 if operation == "finish" and published else 1)
        assert_old_denied_in_fresh_process(owner, old)
        with pytest.raises(CoordinatorRefusal), fresh.guard(old):
            pytest.fail("old receipt resurrected after acknowledged begin")
        if operation == "finish" and published:
            assert disk.pending == ()
            assert_child(owner, "capture", "CAPTURE 2")
            assert (owner[0] / "toy-target").read_bytes() == b"actual intended mutation"
        else:
            assert len(disk.pending) == 1
            assert_child(owner, "capture", "DENIED")
    if operation == "begin":
        assert not (owner[0] / "toy-target").exists()


@pytest.mark.parametrize("stage", ["before-rename", "after-rename"])
def test_completion_process_crash_at_visibility_linearization(owner, stage):
    c = initialized(owner)
    old = c.capture()
    script = _CHILD.split("if action == 'capture':")[0] + """
tx = c.begin_mutation(effect=effect())
target = Path(home) / 'toy-target'
with target.open('wb') as f:
    f.write(b'verified before completion crash')
    f.flush()
    os.fsync(f.fileno())
observed = target.read_bytes()
assert observed == b'verified before completion crash'
verified = m.VerifiedOutcome(tx.effect, hashlib.sha256(observed).hexdigest())
original = os.replace
def replace(src, dst, **kwargs):
    if dst == 'state.json' and action == 'before-rename':
        os._exit(24)
    result = original(src, dst, **kwargs)
    if dst == 'state.json':
        os._exit(25)
    return result
os.replace = replace
c.finish_mutation(tx, verified_outcome=verified)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(owner[0]), str(owner[1]), stage],
        cwd=owner[0],
        env=dict(os.environ, PYTHONPATH=str(Path(module.__file__).parents[2])),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert (result.returncode, result.stdout, result.stderr) == (
        24 if stage == "before-rename" else 25,
        "",
        "",
    )
    fresh = enabled()
    assert_old_denied_in_fresh_process(owner, old)
    with pytest.raises(CoordinatorRefusal), fresh.guard(old):
        pytest.fail("completion crash resurrected pre-begin receipt")
    if stage == "before-rename":
        assert len(state(fresh).pending) == 1
        assert_child(owner, "capture", "DENIED")
    else:
        assert state(fresh).pending == ()
        assert_child(owner, "capture", "CAPTURE 2")


def test_exact_effect_bound_completion_ack_failure_and_readonly_restart_fact(owner, monkeypatch):
    c = initialized(owner)
    old = c.capture()
    tx = c.begin_mutation(effect=effect())
    verified = publish_and_verify(owner)
    original = c._read
    renamed = False
    replace = os.replace

    def mark_rename(src, dst, **kwargs):
        nonlocal renamed
        result = replace(src, dst, **kwargs)
        if dst == "state.json":
            renamed = True
        return result

    def failed_ack(fd, name):
        if renamed and name == "state.json":
            raise OSError("completion committed, exact readback acknowledgement lost")
        return original(fd, name)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", mark_rename)
        patch.setattr(c, "_read", failed_ack)
        with pytest.raises(CoordinatorIndeterminate) as uncertain:
            c.finish_mutation(tx, verified_outcome=verified)
    assert uncertain.value.outcome == MutationOutcome.INDETERMINATE
    for read in (c.capture, lambda: c.inspect_completion(tx, verified_outcome=verified)):
        with pytest.raises(CoordinatorRefusal):
            read()
    fresh = enabled()
    before = {p.name: (p.stat().st_ino, p.read_bytes()) for p in domain(fresh).iterdir()}
    evidence = fresh.inspect_completion(tx, verified_outcome=verified)
    assert evidence.transaction == tx and evidence.verified == verified and evidence.revision == 2
    assert {p.name: (p.stat().st_ino, p.read_bytes()) for p in domain(fresh).iterdir()} == before
    # A new factual observation does NOT mutate/upgrade the prior call or its
    # latched handle, reexecute the toy publication, or authorize a guarded read.
    assert uncertain.value.outcome == MutationOutcome.INDETERMINATE
    with pytest.raises(CoordinatorRefusal), fresh.guard(old):
        pytest.fail("exact completion fact resurrected an old receipt")
    wrong_effect = dataclasses.replace(effect(), boundary_key="profile-B:toy-target")
    wrong_operation = dataclasses.replace(effect(), operation_key="different-operation")
    for wrong_tx, wrong_verified in (
        (dataclasses.replace(tx, transaction=str(uuid4())), verified),
        (dataclasses.replace(tx, incarnation=str(uuid4())), verified),
        (dataclasses.replace(tx, domain="another-domain"), verified),
        (
            dataclasses.replace(tx, effect=wrong_effect),
            dataclasses.replace(verified, effect=wrong_effect),
        ),
        (
            dataclasses.replace(tx, effect=wrong_operation),
            dataclasses.replace(verified, effect=wrong_operation),
        ),
        (
            tx,
            dataclasses.replace(
                verified, fingerprint=hashlib.sha256(b"different-result").hexdigest()
            ),
        ),
    ):
        with pytest.raises(CoordinatorRefusal):
            fresh.inspect_completion(wrong_tx, verified_outcome=wrong_verified)
    # Recognition in an independently started interpreter uses the exact key
    # from the owned application encoder, not reconstructed or fabricated state.
    script = _CHILD.split("if action == 'capture':")[0] + """
import json
payload = json.loads(sys.argv[4])
tx = c._parse_receipt(payload['transaction'])
verified = m.VerifiedOutcome(tx.effect, payload['fingerprint'])
evidence = c.inspect_completion(tx, verified_outcome=verified)
print('EXACT-FACT', evidence.revision,
      evidence.transaction.transaction, evidence.verified.fingerprint)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(owner[0]),
            str(owner[1]),
            "inspect",
            json.dumps({"transaction": c._receipt_data(tx), "fingerprint": verified.fingerprint}),
        ],
        cwd=owner[0],
        env=dict(os.environ, PYTHONPATH=str(Path(module.__file__).parents[2])),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert (result.returncode, result.stdout.strip(), result.stderr) == (
        0,
        f"EXACT-FACT 2 {tx.transaction} {verified.fingerprint}",
        "",
    )
    with pytest.raises(CoordinatorRefusal):
        fresh.finish_mutation(tx, verified_outcome=verified)  # not an idempotent retry API


def test_exact_effect_identity_required_before_finishing(owner):
    c = initialized(owner)
    tx = c.begin_mutation(effect=effect())
    verified = publish_and_verify(owner)
    wrong = dataclasses.replace(
        verified, effect=dataclasses.replace(effect(), boundary_key="profile-B:toy-target")
    )
    with pytest.raises(CoordinatorRefusal) as denied:
        c.finish_mutation(tx, verified_outcome=wrong)
    assert denied.value.outcome == MutationOutcome.DENIED
    assert state(enabled()).pending == (tx,)
    assert {outcome.value for outcome in MutationOutcome} == {"finished", "denied", "indeterminate"}


@pytest.mark.parametrize(
    "corrupt",
    ["partial", "wrong-effect", "wrong-tx", "wrong-fingerprint", "revision", "missing-completion"],
)
def test_closed_completed_records_and_exact_identity_fail_closed(owner, corrupt):
    c = initialized(owner)
    tx = c.begin_mutation(effect=effect())
    verified = publish_and_verify(owner)
    c.finish_mutation(tx, verified_outcome=verified)

    def damage_completed(raw):
        entry = raw["completed"][0]
        if corrupt == "partial":
            del entry["fingerprint"]
        elif corrupt == "wrong-effect":
            entry["transaction"]["effect"]["boundary_key"] = "profile-B:toy-target"
        elif corrupt == "wrong-tx":
            entry["transaction"]["transaction"] = str(uuid4())
        elif corrupt == "wrong-fingerprint":
            entry["fingerprint"] = hashlib.sha256(b"other verified effect").hexdigest()
        elif corrupt == "revision":
            entry["revision"] = True
        else:
            raw["completed"] = []
        return raw

    damage(c, "state.json", damage_completed)
    with pytest.raises(CoordinatorRefusal):
        enabled().inspect_completion(tx, verified_outcome=verified)


@pytest.mark.parametrize("file", ["state.json", "initialized.json", ".mutation.lock"])
@pytest.mark.parametrize("fault", ["missing", "symlink", "mode", "hardlink", "directory"])
def test_file_metadata_identity_fail_closed(owner, file, fault):
    c = initialized(owner)
    root = domain(c)
    path = root / file
    if fault == "mode":
        path.chmod(0o644)
    elif fault == "hardlink":
        os.link(path, root / "extra-link")
    else:
        original = path.read_bytes()
        path.unlink()
        if fault == "symlink":
            other = root / "synthetic-target"
            other.write_bytes(original)
            other.chmod(0o600)
            path.symlink_to(other)
        elif fault == "directory":
            path.mkdir(mode=0o700)
    with pytest.raises(CoordinatorRefusal):
        enabled().capture()


@pytest.mark.parametrize("part", ["home", ".local", "file-release", "domain"])
@pytest.mark.parametrize("fault", ["symlink", "mode"])
def test_directory_symlink_or_unsafe_mode_denies(owner, part, fault):
    c = initialized(owner)
    root = domain(c)
    if part == "home":
        path = owner[0]
    elif part == "domain":
        path = root
    elif part == ".local":
        path = owner[0] / ".local"
    else:
        path = root.parents[1]
    if fault == "mode":
        path.chmod(0o777)
    else:
        moved = path.with_name(path.name + "-moved")
        path.rename(moved)
        path.symlink_to(moved, target_is_directory=True)
    with pytest.raises(CoordinatorRefusal):
        enabled().capture()


@pytest.mark.parametrize("target", ["home", "state.json"])
def test_foreign_owner_metadata_denies_without_repair(owner, monkeypatch, target):
    c = initialized(owner)
    path = owner[0] if target == "home" else domain(c) / target
    real_fstat = os.fstat

    def foreign(fd):
        info = real_fstat(fd)
        if Path(os.readlink(f"/proc/self/fd/{fd}")) == path:
            values = list(info)
            values[4] = os.getuid() + 1
            return os.stat_result(values)
        return info

    monkeypatch.setattr(os, "fstat", foreign)
    with pytest.raises(CoordinatorRefusal):
        enabled().capture()


def test_open_guard_detects_renamed_domain_before_acknowledgement(owner):
    c = initialized(owner)
    root = domain(c)
    old = c.capture()
    with pytest.raises(CoordinatorRefusal):
        with c.guard(old):
            root.rename(root.with_name(root.name + "-old"))
            root.mkdir(mode=0o700)
    with pytest.raises(CoordinatorRefusal):
        c.capture()
    with pytest.raises(CoordinatorRefusal):
        enabled().capture()


def test_replaced_lock_inode_stale_handle_and_replaced_domain(owner):
    c = initialized(owner)
    path = domain(c) / ".mutation.lock"
    path.rename(path.with_name("old-lock"))
    path.write_bytes(b"")
    path.chmod(0o600)
    with pytest.raises(CoordinatorRefusal):
        c.capture()
    with pytest.raises(CoordinatorRefusal):
        enabled().capture()  # initialization witness pins the original inode


@pytest.mark.parametrize(
    "fault",
    [
        "extra",
        "schema",
        "bool-revision",
        "negative-revision",
        "duplicate-pending",
        "bad-tx",
        "incarnation",
        "duplicate-key",
        "noncanonical",
        "truncated",
    ],
)
def test_closed_state_schema_corruption_and_ambiguity(owner, fault):
    c = initialized(owner)
    path = domain(c) / "state.json"
    if fault == "duplicate-key":
        path.write_bytes(b'{"schema":1,"schema":1}')
    elif fault == "noncanonical":
        path.write_text(json.dumps(json.loads(path.read_bytes()), indent=2))
    elif fault == "truncated":
        path.write_bytes(b"{")
    else:

        def corrupt(raw):
            if fault == "extra":
                raw["extra"] = 1
            elif fault == "schema":
                raw["schema"] = True
            elif fault == "bool-revision":
                raw["revision"] = False
            elif fault == "negative-revision":
                raw["revision"] = -1
            elif fault == "incarnation":
                raw["incarnation"] = str(uuid4())
            elif fault == "bad-tx":
                raw["pending"] = ["not-a-transaction"]
            else:
                tx = str(uuid4())
                raw["pending"] = [tx, tx]
            return raw

        damage(c, "state.json", corrupt)
    with pytest.raises(CoordinatorRefusal):
        enabled().capture()


@pytest.mark.parametrize("fault", ["incarnation", "partial", "schema", "lock-identity"])
def test_initialization_witness_mismatch_partial_or_unknown_denies(owner, fault):
    c = initialized(owner)
    old = c.capture()
    tx = c.begin_mutation(effect=effect())
    assert tx in state(c).pending

    def corrupt(raw):
        if fault == "incarnation":
            raw["incarnation"] = str(uuid4())
        elif fault == "partial":
            del raw["lock_inode"]
        elif fault == "schema":
            raw["schema"] = True
        else:
            raw["lock_inode"] += 1
        return raw

    damage(c, "initialized.json", corrupt)
    with pytest.raises(CoordinatorRefusal):
        enabled().capture()
    assert_old_denied_in_fresh_process(owner, old)
    with pytest.raises(CoordinatorRefusal), enabled().guard(old):
        pytest.fail("partial/mismatched witness accepted an old receipt")


def test_stale_incarnation_domain_handles_and_exact_receipt_types(owner):
    c = initialized(owner)
    receipt = c.capture()
    for altered in (
        dataclasses.replace(receipt, incarnation=str(uuid4())),
        dataclasses.replace(receipt, domain="different-domain"),
        dataclasses.asdict(receipt),
    ):
        with pytest.raises(CoordinatorRefusal), c.guard(cast(Any, altered)):
            pytest.fail("stale/structural receipt accepted")
    with pytest.raises(TypeError):
        dataclasses.replace(receipt, revision=False)
    tx = c.begin_mutation(effect=effect())
    for altered in (
        dataclasses.replace(tx, incarnation=str(uuid4())),
        dataclasses.replace(tx, domain="different-domain"),
        dataclasses.replace(tx, transaction=str(uuid4())),
    ):
        with pytest.raises(CoordinatorRefusal):
            enabled().finish_mutation(altered, verified_outcome=publish_and_verify(owner))
    assert state(enabled()).pending == (tx,)


@contextlib.contextmanager
def tcp_pair():
    listener = socket.socket()
    sender = socket.socket()
    receiver = None
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(5)
        sender.settimeout(5)
        sender.connect(listener.getsockname())
        receiver, _ = listener.accept()
        sender.setblocking(False)
        receiver.settimeout(5)
        yield sender, receiver
    finally:
        if receiver is not None:
            receiver.close()
        sender.close()
        listener.close()


def test_real_nonblocking_tcp_one_emission_then_process_mutation_cancels_continuation(owner):
    c = initialized(owner)
    receipt = c.capture()  # BEFORE hypothetical slow independent verification
    with tcp_pair() as (sender, receiver):
        with c.guard(receipt) as observation:
            assert observation.current is True
            accepted = sender.send(b"sealed-prefix", socket.MSG_DONTWAIT)
        assert accepted == len(b"sealed-prefix")
        assert receiver.recv(accepted) == b"sealed-prefix"
        assert_child(owner, "mutate", "COMMITTED")
        with pytest.raises(CoordinatorRefusal), c.guard(receipt):
            sender.send(b"forbidden-continuation", socket.MSG_DONTWAIT)
        assert not select.select([receiver], [], [], 0)[0]


def _consumer_references(source):
    if "file_release_coordinator" not in source and "FileReleaseCoordinator" not in source:
        return []
    references = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            names = [node.module or ""] + [alias.name for alias in node.names]
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names = [node.value]
        elif isinstance(node, ast.Name):
            names = [node.id]
        elif isinstance(node, ast.Attribute):
            names = [node.attr]
        else:
            continue
        if any(
            "file_release_coordinator" in name or "FileReleaseCoordinator" in name for name in names
        ):
            references.append(node.lineno)
    return references


def test_no_production_consumer_or_environment_override():
    own = Path(module.__file__).resolve()
    root = own.parents[1]
    violations = []
    for path in root.rglob("*.py"):
        if path.resolve() != own:
            violations.extend(
                (str(path.relative_to(root)), line)
                for line in _consumer_references(path.read_text(encoding="utf-8"))
            )
    assert violations == []
    source = own.read_text(encoding="utf-8")
    assert "os.getenv(" not in source and "os.environ" not in source
    assert "linux_prepared_writer" not in source
    assert "LinuxPreparedRecordWriter" not in source


@pytest.mark.parametrize(
    "source",
    [
        "from local_operator.network.file_release_coordinator import FileReleaseCoordinator as C",
        "from . import file_release_coordinator as m",
        "import local_operator.network.file_release_coordinator as m",
        'importlib.import_module("local_operator.network.file_release_coordinator")',
        "module.FileReleaseCoordinator()",
    ],
)
def test_dormancy_scanner_negative_controls(source):
    assert _consumer_references(source)


def test_reverse_process_mutation_before_guard_denies_all_tcp_bytes(owner):
    c = initialized(owner)
    receipt = c.capture()
    assert_child(owner, "mutate", "COMMITTED")
    with tcp_pair() as (sender, receiver):
        with pytest.raises(CoordinatorRefusal), c.guard(receipt):
            sender.send(b"forbidden-first-byte", socket.MSG_DONTWAIT)
        assert not select.select([receiver], [], [], 0)[0]
