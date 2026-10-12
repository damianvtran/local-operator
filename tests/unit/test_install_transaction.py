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
from pathlib import Path
from typing import Any

import pytest

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
