"""Owned roots only: transactions, observational status and delayed promotion.

The installer seam creates actual dist-info files. Publication fixtures model
channel answers explicitly; they are not claimed as a real PyPI/signing run.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import tomllib
from pathlib import Path
from typing import Any

import pytest
from packaging.requirements import Requirement

from local_operator import install_transaction as tx
from local_operator import update


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("LOP_INSTALL_ROOT", raising=False)

    def download(generation: Path, built: update.PublicationEvidence) -> Path:
        # Explicit synthetic artifact transport; the real downloader has its
        # own hash/time/size contract tests, not a fabricated uv-hash guarantee.
        artifact = update._candidate_artifact(generation, built)
        artifact.parent.mkdir()
        artifact.write_bytes(b"synthetic-wheel")
        return artifact

    monkeypatch.setattr(update, "_download_candidate_artifact", download)
    return tmp_path


def tiny_tree(argv: list[str], env: dict[str, str]) -> int:
    target = argv[-1].split("==")[-1]
    if target == "local-operator":
        target = Path(argv[argv.index("--from") + 1]).name.split("-")[1]
    root = Path(env["UV_TOOL_DIR"]) / "local-operator"
    dist = (
        root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / f"local_operator-{target}.dist-info"
    )
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: local-operator\nVersion: {target}\n"
    )
    if "--from" in argv:
        source = argv[argv.index("--from") + 1]
        if source.endswith(".whl"):
            (dist / "direct_url.json").write_text(
                json.dumps({"url": Path(source).resolve().as_uri(), "archive_info": {}})
            )
    return 0


def source_runner(target: str):
    def run(argv: list[str], env: dict[str, str]) -> int:
        return tiny_tree([*argv[:-1], f"local-operator=={target}"], env)

    return run


def test_permanent_inode_and_nested_actual_handle(isolated: Path) -> None:
    root = update.stable_root()
    with tx.transaction(root, operation="outer") as outer:
        inode = (root / tx.LOCK_NAME).stat().st_ino
        with tx.transaction(root / ".." / root.name, operation="inner") as inner:
            assert inner is outer
            inner.validate(root)
        outer.validate(root)
        with pytest.raises(tx.InstallTransactionError, match="another root"):
            outer.validate(root / "other")
    with tx.transaction(root, operation="next"):
        assert (root / tx.LOCK_NAME).stat().st_ino == inode
    with pytest.raises(tx.InstallTransactionError, match="no longer holds"):
        outer.validate(root)


def test_absent_status_does_not_create_root(
    isolated: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = update.stable_root()
    assert tx.status(root)["os_owners"] == []
    assert update.install_status_command() == 0
    assert "install transaction:" in capsys.readouterr().out
    assert not root.exists()


def test_status_preserves_record_inode_and_content(isolated: Path) -> None:
    root = update.stable_root()
    with tx.transaction(root, operation="holder"):
        path = root / tx.LOCK_NAME
        before = (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes())
        observed = tx.status(root)
        assert observed["metadata"]["pid"] == os.getpid()
        # Kernel inspection may not be available on every CI platform. Unknown
        # must never be upgraded to a claimed owner from live metadata.
        assert observed["os_owners"] is None or os.getpid() in observed["os_owners"]
        assert before == (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes())
    observed = tx.status(root)
    assert observed["os_owners"] in (None, [])
    assert observed["metadata_alive"] is True


def test_real_subprocess_contention_across_config_profiles(isolated: Path) -> None:
    source = Path(update.__file__).resolve().parent.parent
    child = """from pathlib import Path
import sys
from local_operator.install_transaction import transaction
with transaction(Path(sys.argv[1]), operation='child', timeout=0):
 print('held', flush=True)
 sys.stdin.readline()
"""
    env = {
        "HOME": str(isolated),
        "LOCAL_OPERATOR_CONFIG_DIR": str(isolated / "other-profile"),
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(source),
    }
    process = subprocess.Popen(
        [sys.executable, "-c", child, str(update.stable_root())],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "held"
        with pytest.raises(tx.InstallTransactionError, match="another install"):
            with tx.transaction(update.stable_root(), operation="contender", timeout=0):
                pytest.fail("distinct config profile bypassed root lock")
        observed = tx.status(update.stable_root())
        assert observed["os_owners"] is None or process.pid in observed["os_owners"]
        process.communicate("release\n", timeout=10)
        assert process.returncode == 0
        with tx.transaction(update.stable_root(), operation="after exit", timeout=0):
            pass
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)


def test_owner_pid_reuse_and_unknown_are_distinct(monkeypatch: pytest.MonkeyPatch) -> None:
    record = {"pid": 123, "birth_scheme": "test", "birth_token": "old"}
    monkeypatch.setattr(tx.procstate, "pid_liveness", lambda _pid: True)
    monkeypatch.setattr(tx.procstate, "same_birth", lambda *_args: False)
    assert tx.owner_alive(record) is False
    monkeypatch.setattr(tx.procstate, "same_birth", lambda *_args: None)
    assert tx.owner_alive(record) is None


def test_live_candidate_survives_marker_age_and_prune(isolated: Path) -> None:
    generation, nonce, _baseline = update._register_candidate("old", "0.70.7", "source")
    root = update._generation_install_root(generation)
    root.mkdir(parents=True)
    update.write_source_marker(root, version="0.70.7")
    plan = update.prune_generations(keep=0, now=10**12)
    assert generation.is_dir()
    assert any("install attempt" in item.reason for item in plan.decisions)
    assert not update._remove_tree(generation)
    update._discard_candidate(generation, nonce)
    assert not generation.exists()


def test_duplicate_target_refuses_before_second_build(isolated: Path) -> None:
    generation, nonce, _baseline = update._register_candidate("A", "0.70.7", "exact")
    with pytest.raises(update.UpdateError, match="another candidate"):
        update._register_candidate("B", "0.70.7", "exact")
    update._discard_candidate(generation, nonce)


def test_delayed_source_candidate_cannot_regress_new_current(isolated: Path) -> None:
    update.install_into_generation(isolated, version="0.70.6", runner=source_runner("0.70.6"))
    started = threading.Event()
    release = threading.Event()
    failures: list[str] = []

    def slow(argv: list[str], env: dict[str, str]) -> int:
        tiny_tree([*argv[:-1], "local-operator==0.70.7"], env)
        started.set()
        assert release.wait(10)
        return 0

    def actor() -> None:
        try:
            update.install_into_generation(isolated, version="0.70.7", runner=slow)
        except update.UpdateError as exc:
            failures.append(str(exc))

    thread = threading.Thread(target=actor)
    thread.start()
    try:
        assert started.wait(10)
        update.install_into_generation(isolated, version="0.70.8", runner=source_runner("0.70.8"))
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive()
    assert failures and "refusing stale candidate" in failures[0]
    assert update.generation_version(update.current_generation()) == "0.70.8"
    assert not any(path.name.endswith("0.70.7") for path in update.generations_dir().iterdir())


def publication_document(target: str) -> dict[str, Any]:
    return {
        "info": {"version": target},
        "urls": [
            {
                "filename": f"local_operator-{target}-py3-none-any.whl",
                "url": f"https://files.pythonhosted.org/packages/{target}.whl",
                "yanked": False,
                "digests": {"sha256": hashlib.sha256(b"synthetic-wheel").hexdigest()},
            }
        ],
    }


def test_publication_wheel_selection_has_declared_runtime_dependency() -> None:
    """A dev-tool transitive must not hide a broken bare PyPI updater install."""
    project = Path(__file__).resolve().parents[2] / "pyproject.toml"
    dependencies = tomllib.loads(project.read_text())["project"]["dependencies"]
    requirements = [Requirement(dependency) for dependency in dependencies]
    # Wheel parsing and host tag selection run before promotion on every OS.
    # Extras or platform markers would still leave some default installs broken.
    assert any(
        requirement.name == "packaging" and requirement.marker is None and not requirement.extras
        for requirement in requirements
    ), "_fresh_publication imports packaging; declare it for every default install"


def test_exact_publication_does_not_silently_become_latest(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(update, "_HOST_IS_MACOS", False)
    monkeypatch.setattr(
        update, "_pypi_release_document", lambda target: (publication_document(target), "")
    )
    monkeypatch.setattr(update, "_fetch_pypi_version", lambda: "0.70.8")
    evidence = update._fresh_publication("0.70.7", "exact")
    assert evidence is not None and evidence.target == "0.70.7"
    with pytest.raises(update.UpdateError, match="fresh PyPI latest"):
        update._fresh_publication("0.70.7", "latest")


def test_unreachable_publication_does_not_use_cache(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(update, "_pypi_release_document", lambda _target: (None, "unreachable"))
    with pytest.raises(update.UpdateError, match="cannot freshly confirm"):
        update.install_into_generation(version="0.70.7", runner=tiny_tree)
    assert not update.stable_root().exists()


def test_installer_pins_artifact_digest_and_rechecks_publication(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(update, "_HOST_IS_MACOS", False)
    calls: list[str] = []

    def document(target: str):
        calls.append(target)
        return publication_document(target), ""

    monkeypatch.setattr(update, "_pypi_release_document", document)
    seen: list[list[str]] = []

    def run(argv: list[str], env: dict[str, str]) -> int:
        seen.append(argv)
        return tiny_tree(argv, env)

    update.install_into_generation(version="0.70.7", runner=run)
    assert calls == ["0.70.7", "0.70.7"]
    assert seen[0][-1] == "local-operator"
    assert "local_operator-0.70.7-py3-none-any.whl" in seen[0][seen[0].index("--from") + 1]
    assert update.generation_version(update.current_generation()) == "0.70.7"


def test_changed_artifact_refuses_promotion_and_cleans_own_candidate(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(update, "_HOST_IS_MACOS", False)
    count = 0

    def document(target: str):
        nonlocal count
        count += 1
        value = publication_document(target)
        if count > 1:
            value["urls"][0]["digests"]["sha256"] = "b" * 64
        return value, ""

    monkeypatch.setattr(update, "_pypi_release_document", document)
    with pytest.raises(update.UpdateError, match="artifact changed"):
        update.install_into_generation(version="0.70.7", runner=tiny_tree)
    assert update.current_generation() is None
    assert not list(update.generations_dir().iterdir())


_REAL_DOWNLOAD = update._download_candidate_artifact


def test_real_downloader_checks_hash_before_uv(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import contextmanager

    import httpx

    @contextmanager
    def stream(method: str, url: str, **kwargs: Any):
        assert method == "GET" and kwargs["follow_redirects"] is False

        class Response:
            def raise_for_status(self) -> None:
                pass

            def iter_bytes(self):
                yield b"synthetic-wheel"

        yield Response()

    monkeypatch.setattr(httpx, "stream", stream)
    candidate = isolated / "candidate"
    candidate.mkdir()
    evidence = update.PublicationEvidence(
        "0.70.7",
        "exact",
        "local_operator-0.70.7-py3-none-any.whl",
        "https://files.pythonhosted.org/test.whl",
        hashlib.sha256(b"synthetic-wheel").hexdigest(),
        0,
    )
    path = _REAL_DOWNLOAD(candidate, evidence)
    assert path.read_bytes() == b"synthetic-wheel"
    other = isolated / "wrong"
    other.mkdir()
    from dataclasses import replace

    with pytest.raises(update.UpdateError, match="SHA256 disagrees"):
        _REAL_DOWNLOAD(other, replace(evidence, digest="0" * 64))
    assert not list((other / "artifact").iterdir())


def test_real_downloader_size_bound_is_fail_closed(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import contextmanager

    import httpx

    @contextmanager
    def stream(*args: Any, **kwargs: Any):
        class Response:
            def raise_for_status(self) -> None:
                pass

            def iter_bytes(self):
                yield b"too-large"

        yield Response()

    monkeypatch.setattr(httpx, "stream", stream)
    monkeypatch.setattr(update, "_ARTIFACT_MAX_BYTES", 3)
    candidate = isolated / "candidate"
    candidate.mkdir()
    evidence = update.PublicationEvidence(
        "0.70.7", "exact", "tiny.whl", "https://files.pythonhosted.org/test.whl", "a" * 64, 0
    )
    with pytest.raises(update.UpdateError, match="time/size bound"):
        _REAL_DOWNLOAD(candidate, evidence)
    assert not list((candidate / "artifact").iterdir())


def test_same_published_target_registration_refuses_latest_exact_duplicate(isolated: Path) -> None:
    candidate, nonce, _ = update._register_candidate("0.70.7", "0.70.7", "latest")
    with pytest.raises(update.UpdateError, match="another candidate"):
        update._register_candidate("0.70.7", "0.70.7", "exact")
    assert tx.attempt_protected(candidate)
    update._discard_candidate(candidate, nonce)
    assert not candidate.exists()


def test_real_busy_record_refuses_before_build_and_again_before_promotion(isolated: Path) -> None:
    from local_operator.session.runtime import registry
    from local_operator.session.runtime.types import SessionRecord

    root = isolated / "config"
    record = SessionRecord(
        pid=os.getpid(),
        kind="daemon",
        session_id="synthetic-busy-proof",
        conversation_name="synthetic",
        cwd=str(isolated),
        model_label="fixture",
        control_port=0,
        control_key="fixture-only",
        busy=True,
    )
    path = registry.publish(record, root)
    called = False

    def run(argv: list[str], env: dict[str, str]) -> int:
        nonlocal called
        called = True
        return source_runner("0.70.7")(argv, env)

    with pytest.raises(update.UpdateError, match="sessions are busy"):
        update.install_into_generation(isolated, version="0.70.7", runner=run)
    assert not called and path.exists()
    path.unlink()

    def become_busy(argv: list[str], env: dict[str, str]) -> int:
        code = run(argv, env)
        registry.publish(record, root)
        return code

    with pytest.raises(update.UpdateError, match="sessions are busy"):
        update.install_into_generation(isolated, version="0.70.7", runner=become_busy)
    assert called and update.current_generation() is None
    assert not list(update.generations_dir().iterdir())


def test_unparseable_session_namespace_is_not_idle(isolated: Path) -> None:
    from local_operator.session.runtime import registry

    directory = registry.ensure_run_dir(isolated / "config")
    bad = directory / "12345.json"
    bad.write_text("{broken")
    with pytest.raises(update.UpdateError, match="cannot prove fleet idle"):
        update.install_into_generation(isolated, version="0.70.7", runner=source_runner("0.70.7"))
    assert bad.read_text() == "{broken"


def test_lock_identity_nonposix_refuses_before_any_filesystem_access(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = update.stable_root()

    def forbidden(*args: Any, **kwargs: Any):
        pytest.fail("unsupported identity validation touched the filesystem")

    with monkeypatch.context() as patch:
        patch.setattr(tx.os, "name", "nt")
        patch.setattr(tx.os, "fstat", forbidden)
        patch.delattr(tx.os, "getuid")  # the actual missing Windows capability
        with pytest.raises(tx.LockUnsupported, match="requires POSIX"):
            tx._validate_lock_identity(root, -1, -1)
    assert not root.exists()


def test_wrong_owned_lock_refuses_before_metadata_write(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import stat

    root = update.stable_root()
    root.mkdir(parents=True)
    lock = root / tx.LOCK_NAME
    lock.write_text("KEEP-LOCK-BYTES")
    real = os.fstat

    def foreign_regular(fd: int):
        sample = real(fd)
        if stat.S_ISREG(sample.st_mode):
            values = list(sample)
            values[4] = sample.st_uid + 1  # labelled ownership observation seam
            return os.stat_result(values)
        return sample

    monkeypatch.setattr(tx.os, "fstat", foreign_regular)
    with pytest.raises(tx.InstallTransactionError, match="unsafe or changed"):
        with tx.transaction(root, operation="foreign lock"):
            pytest.fail("foreign-owned lock accepted")
    assert lock.read_text() == "KEEP-LOCK-BYTES"


@pytest.mark.parametrize("shape", ["symlink", "directory", "fifo", "hardlink", "writable"])
def test_unsafe_existing_lock_shape_preserves_external_bytes(isolated: Path, shape: str) -> None:
    root = update.stable_root()
    root.mkdir(parents=True)
    lock = root / tx.LOCK_NAME
    sentinel = isolated / "sentinel"
    sentinel.write_text("KEEP-USER-BYTES")
    if shape == "symlink":
        lock.symlink_to(sentinel)
    elif shape == "directory":
        lock.mkdir()
    elif shape == "fifo":
        os.mkfifo(lock)
    elif shape == "hardlink":
        os.link(sentinel, lock)
    else:
        lock.write_text("KEEP-LOCK-BYTES")
        lock.chmod(0o666)
    with pytest.raises(tx.InstallTransactionError):
        with tx.transaction(root, operation="unsafe fixture", timeout=0):
            pytest.fail("unsafe lock accepted")
    assert sentinel.read_text() == "KEEP-USER-BYTES"
    print(f"ACTUAL unsafe lock {shape} refused; external sentinel unchanged")
    if shape == "writable":
        assert lock.read_text() == "KEEP-LOCK-BYTES"


def test_wrong_owned_root_refuses_before_lock_creation(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = update.stable_root()
    root.mkdir(parents=True)
    owner = os.getuid()
    monkeypatch.setattr(tx.os, "getuid", lambda: owner + 1)
    with pytest.raises(tx.InstallTransactionError, match="owned directory"):
        with tx.transaction(root, operation="foreign owner"):
            pytest.fail("foreign-owned directory accepted")
    assert not (root / tx.LOCK_NAME).exists()


def test_root_replacement_after_flock_refuses_before_metadata_write(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fcntl

    root = update.stable_root()
    root.mkdir(parents=True)
    lock = root / tx.LOCK_NAME
    lock.write_text("KEEP-LOCK-BYTES")
    moved = root.with_name("detached-root")
    real = fcntl.flock
    swapped = False

    def replace_after_acquire(fd: int, mode: int) -> None:
        nonlocal swapped
        real(fd, mode)
        if mode & fcntl.LOCK_EX and not swapped:
            swapped = True
            root.rename(moved)
            root.mkdir()
            (root / tx.LOCK_NAME).write_text("KEEP-NEW-LOCK-BYTES")

    monkeypatch.setattr(fcntl, "flock", replace_after_acquire)
    with pytest.raises(tx.InstallTransactionError, match="changed install lock/root"):
        with tx.transaction(root, operation="root race"):
            pytest.fail("changed root accepted")
    assert (moved / tx.LOCK_NAME).read_text() == "KEEP-LOCK-BYTES"
    assert (root / tx.LOCK_NAME).read_text() == "KEEP-NEW-LOCK-BYTES"


def test_identical_exact_artifact_reuses_current_without_second_builder(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(update, "_HOST_IS_MACOS", False)
    monkeypatch.setattr(
        update, "_pypi_release_document", lambda target: (publication_document(target), "")
    )
    builds = 0

    def build(argv: list[str], env: dict[str, str]) -> int:
        nonlocal builds
        builds += 1
        return tiny_tree(argv, env)

    first = update.install_into_generation(version="0.70.7", runner=build)
    second = update.install_into_generation(version="0.70.7", runner=build)
    assert first == second == update.current_generation()
    assert builds == 1
    print(f"ACTUAL identical exact reuse same={first == second}, builder_calls={builds}")
    assert len(list(update.generations_dir().iterdir())) == 1


def test_actual_mesh_capture_bound_tracks_worker_and_preserves_failure_output(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.network.meshupdate import _CapturedInstall

    candidate, nonce, _ = update._register_candidate("mesh-bound", "0.70.7", "source")
    real = tx.run_installer
    monkeypatch.setattr(
        tx, "run_installer", lambda argv, **kwargs: real(argv, timeout=2.0, **kwargs)
    )
    capture = _CapturedInstall()
    with pytest.raises(tx.InstallTransactionError, match="exceeded"):
        capture(
            [
                sys.executable,
                "-c",
                "import sys,time; print('mesh failure evidence',flush=True);time.sleep(60)",
            ],
            update._generation_env(candidate),
        )
    record = tx.read_record(candidate / tx.ATTEMPT_NAME)
    assert record is not None and isinstance(record.get("worker"), dict)
    assert "mesh failure evidence" in capture.output
    print(
        f"ACTUAL mesh bound worker={record['worker']['pid']} "
        f"alive={tx.owner_alive(record['worker'])} output={capture.output!r}"
    )
    assert tx.owner_alive(record["worker"]) is False
    update._discard_candidate(candidate, nonce)


def test_actual_mesh_capture_cancel_reaps_owned_worker(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.network.meshupdate import _CapturedInstall

    candidate, nonce, _ = update._register_candidate("mesh-cancel", "0.70.7", "source")
    real = subprocess.Popen.wait
    cancelled = False

    def cancel_once(process, *args, **kwargs):
        nonlocal cancelled
        if process.args[0] == sys.executable and not cancelled:
            cancelled = True
            raise KeyboardInterrupt("owned fixture cancellation")
        return real(process, *args, **kwargs)

    monkeypatch.setattr(subprocess.Popen, "wait", cancel_once)
    with pytest.raises(KeyboardInterrupt):
        _CapturedInstall()(
            [sys.executable, "-c", "import time;time.sleep(60)"], update._generation_env(candidate)
        )
    record = tx.read_record(candidate / tx.ATTEMPT_NAME)
    assert record is not None and tx.owner_alive(record["worker"]) is False
    print(f"ACTUAL mesh cancellation worker={record['worker']['pid']} alive=False")
    update._discard_candidate(candidate, nonce)


def test_actual_mesh_capture_owner_crash_keeps_live_worker_candidate(isolated: Path) -> None:
    import signal
    import time

    source = Path(update.__file__).resolve().parent.parent
    script = """from pathlib import Path
import sys
from local_operator import update
from local_operator.network.meshupdate import _CapturedInstall
candidate,_,_=update._register_candidate('mesh-crash','0.70.7','source')
print(str(candidate),flush=True)
_CapturedInstall()([sys.executable,'-c','import time;time.sleep(60)'],
                   update._generation_env(candidate))
"""
    env = {
        "HOME": str(isolated),
        "LOCAL_OPERATOR_CONFIG_DIR": str(isolated / "child-config"),
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(source),
    }
    owner = subprocess.Popen(
        [sys.executable, "-c", script],
        env=env,
        cwd=isolated,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    worker = None
    try:
        assert owner.stdout is not None
        candidate = Path(owner.stdout.readline().strip())
        assert candidate.is_relative_to(isolated)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            record = tx.read_record(candidate / tx.ATTEMPT_NAME)
            if record and record.get("worker"):
                worker = record["worker"]["pid"]
                break
            time.sleep(0.01)
        assert worker is not None
        os.kill(owner.pid, signal.SIGKILL)
        owner.wait(timeout=3)
        assert tx.attempt_protected(candidate)
        plan = update.prune_generations(keep=0, referenced=(), now=10**12)
        assert candidate.exists() and not plan.removed
        print(
            f"ACTUAL crashed mesh owner={owner.pid} worker={worker} "
            f"protected={tx.attempt_protected(candidate)} "
            f"candidate_exists={candidate.exists()} prune_removed={len(plan.removed)}"
        )
    finally:
        if worker:
            try:
                os.killpg(worker, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if owner.poll() is None:
            os.killpg(owner.pid, signal.SIGKILL)
            owner.wait(timeout=3)


def test_actual_member_handler_reports_superseded_tail_current_and_serving(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from local_operator.network import meshupdate as mesh

    monkeypatch.setattr(update, "_HOST_IS_MACOS", False)
    monkeypatch.setattr(
        update, "_pypi_release_document", lambda target: (publication_document(target), "")
    )
    update.install_into_generation(isolated, version="0.70.6", runner=source_runner("0.70.6"))
    monkeypatch.setattr(mesh, "detect_install_kind", lambda: update.InstallKind.UV_TOOL)
    monkeypatch.setattr(
        mesh,
        "current_member_version",
        lambda: update.generation_version(update.current_generation()),
    )
    monkeypatch.setattr(
        mesh, "check_published", lambda target: (True, "controlled publication dependency")
    )
    monkeypatch.setattr(mesh, "_audit_member", lambda *args, **kwargs: None)
    monkeypatch.setattr(mesh, "roll_services_after_reply", lambda: None)
    monkeypatch.setattr(mesh, "reengage_displaced", lambda *args, **kwargs: (0, ""))

    def supersede(*args, **kwargs):
        update.install_into_generation(version="0.70.8", runner=tiny_tree)
        return [], "controlled external convergence seam"

    monkeypatch.setattr(mesh, "refresh_runtimes", supersede)
    # Minimal controlled context: actual handler body, not a serving/network proof.
    server: Any = SimpleNamespace(root=isolated / "config")
    result = mesh.member_execute(
        server,
        None,
        {"target": {"version": "0.70.7", "source_ref": ""}},
        runner=tiny_tree,
    )
    assert result["state"] == "done"
    assert result["promoted"] == "0.70.7"
    assert result["now_current"] == result["version"] == "0.70.8"
    assert result["serving"] == mesh.origin_target()["version"]
    assert update.generation_version(update.current_generation()) == "0.70.8"
    print("ACTUAL superseded member receipt " + json.dumps(result, sort_keys=True))
