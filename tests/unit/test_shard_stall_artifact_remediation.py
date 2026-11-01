"""Review reproductions: real CLI and workflow boundary, synthetic file data only."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests import shard_stall_artifact as artifact

RUN = "ci-3293-1-3.12-0"
REPO = Path(__file__).parents[2]


def cli(root, output, registry):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.shard_stall_artifact",
            "--root",
            str(root),
            "--invocation",
            RUN,
            "--registry",
            str(registry),
            "--output",
            str(output),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        timeout=10,
    )


@pytest.mark.parametrize("failure", ["invalid-identity", "source-error"])
def test_r1_failed_stage_never_authorizes_upload_of_preexisting_sentinel(tmp_path, failure):
    root = tmp_path / "root"
    (root / RUN).mkdir(parents=True)
    registry = tmp_path / "owners"
    registry.mkdir()
    old = tmp_path / "lo-stall-artifact"
    old.mkdir()
    (old / "private-fixture.txt").write_text("must never upload")
    (old / "manifest.json").write_text('{"invocation":"STALE"}')
    # Actual CLI refuses unknown existing data, without deleting or rewriting it.
    result = cli(root, old, registry)
    assert result.returncode == 2
    assert (old / "private-fixture.txt").read_text() == "must never upload"
    workflow = yaml.safe_load((REPO / ".github/workflows/ci.yml").read_text())
    steps = workflow["jobs"]["test"]["steps"]
    stage = next(s for s in steps if s.get("id") == "stage_stall")
    upload = next(s for s in steps if s.get("name") == "Upload current-invocation timeout files")
    assert "steps.stage_stall.outcome == 'success'" in upload["if"]
    assert "steps.stage_stall.outputs.artifact != ''" in upload["if"]
    assert upload["with"]["path"] == "${{ steps.stage_stall.outputs.artifact }}"
    command = stage["run"].replace('"${TMPDIR:-/tmp}/lo-shard-stall"', '"' + str(root) + '"')
    # Fail the real collector through its actual invocation protocol after fresh
    # allocation. A failure must not write any path to the workflow output file.
    outputs = tmp_path / "outputs"
    env = {
        **os.environ,
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_OUTPUT": str(outputs),
        artifact.ENV_INVOCATION: "../invalid" if failure == "invalid-identity" else RUN,
        artifact.ENV_REGISTRY: str(registry),
    }
    if failure == "source-error":
        wrong_type = tmp_path / "wrong-type"
        wrong_type.write_text("not a source directory")
        command = command.replace(str(root), str(wrong_type))
    failed = subprocess.run(
        ["bash", "-eu", "-c", command],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert failed.returncode != 0
    assert not outputs.exists() or not outputs.read_text().strip()
    assert (old / "private-fixture.txt").read_text() == "must never upload"


def test_r2_no_worker_registration_means_no_numeric_impostor_export(tmp_path):
    root = tmp_path / "root"
    run = root / RUN
    run.mkdir(parents=True)
    registry = tmp_path / "owners"
    registry.mkdir()
    (run / "worker-999999.log").write_text("SYNTHETIC-UNRELATED-FIXTURE-CONTENT")
    result = cli(root, tmp_path / "out", registry)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "out/manifest.json").read_text())
    assert manifest["status"] == "missing"
    assert manifest["files"] == []
    assert not (tmp_path / "out/worker-999999.log").exists()


def test_r2_registered_writer_excludes_impostor_and_path_replacement(tmp_path):
    root = tmp_path / "root"
    run = root / RUN
    run.mkdir(parents=True)
    registry = tmp_path / "owners"
    registry.mkdir()
    owned = run / "worker-42.log"
    with artifact.registered_handle(owned, registry) as handle:
        handle.write("Timeout (0:04:00)!\\nactual writer\\n")
    (run / "worker-999999.log").write_text("ordinary fixture impostor")
    result = cli(root, tmp_path / "out", registry)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out/worker-42.log").exists()
    assert not (tmp_path / "out/worker-999999.log").exists()
    # Keep the old inode alive so the filesystem cannot reuse it for the impostor.
    owned.rename(run / "saved-inode")
    owned.write_text("replacement fixture content")
    result = cli(root, tmp_path / "replaced", registry)
    assert result.returncode == 2
    manifest = json.loads((tmp_path / "replaced/manifest.json").read_text())
    assert manifest["files"][0]["reason"] == "writer-inode-mismatch"
    assert not (tmp_path / "replaced/worker-42.log").exists()


def test_r2_lifecycle_registration_before_tests_and_consumed_handoff(tmp_path, monkeypatch):
    from tests import shard_stall_watchdog as watchdog

    root = tmp_path / "root"
    run = root / RUN
    run.mkdir(parents=True)
    registry = tmp_path / "owners"
    registry.mkdir()
    monkeypatch.setattr(watchdog, "_RUN_DIR", run)
    monkeypatch.setenv(artifact.ENV_REGISTRY, str(registry))
    timer = watchdog._WorkerTimer(240)
    try:
        assert artifact.ENV_REGISTRY not in os.environ
        assert timer._handle is not None
        assert not timer._armed  # Only file preparation: no timer/native dispatch.
        timer._handle.write("synthetic writer data")
        timer._handle.flush()
        manifest = artifact.stage(root, RUN, tmp_path / "out", registry=registry)
        assert manifest["files"][0]["status"] == "retained"
        assert manifest["output_bytes"] == len("synthetic writer data")
    finally:
        if timer._handle is not None:
            timer._handle.close()


def test_r3_read_errors_and_partial_totals_fail_closed(tmp_path, monkeypatch):
    import errno

    root = tmp_path / "root"
    run = root / RUN
    run.mkdir(parents=True)
    registry = tmp_path / "owners"
    registry.mkdir()
    for pid in (42, 43):
        with artifact.registered_handle(run / f"worker-{pid}.log", registry) as handle:
            handle.write("SYNTHETIC")
    real_open, real_read = artifact.os.open, artifact.os.read
    fail_fd = set()

    def opened(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        if path == "worker-43.log":
            fail_fd.add(fd)
        else:
            fail_fd.discard(fd)
        return fd

    def read(fd, amount):
        if fd in fail_fd:
            raise OSError(errno.EIO, "synthetic read failure")
        return real_read(fd, amount)

    monkeypatch.setattr(artifact.os, "open", opened)
    monkeypatch.setattr(artifact.os, "read", read)
    assert (
        artifact.main(
            [
                "--root",
                str(root),
                "--invocation",
                RUN,
                "--registry",
                str(registry),
                "--output",
                str(tmp_path / "out"),
            ]
        )
        == 2
    )
    manifest = json.loads((tmp_path / "out/manifest.json").read_text())
    assert manifest["status"] == "partial"
    assert manifest["output_bytes"] == 9
    assert [f["status"] for f in manifest["files"]] == ["retained", "error"]
    assert manifest["files"][1]["output_bytes"] == 0
    assert not (tmp_path / "out/worker-43.log").exists()
    (run / "worker-42.log").unlink()
    assert (
        artifact.main(
            [
                "--root",
                str(root),
                "--invocation",
                RUN,
                "--registry",
                str(registry),
                "--output",
                str(tmp_path / "error"),
            ]
        )
        == 2
    )
    manifest = json.loads((tmp_path / "error/manifest.json").read_text())
    assert manifest["status"] == "error"
    assert manifest["output_bytes"] == 0


def test_registered_source_growth_is_bounded_and_reported(tmp_path, monkeypatch):
    root = tmp_path / "root"
    run = root / RUN
    run.mkdir(parents=True)
    registry = tmp_path / "owners"
    registry.mkdir()
    path = run / "worker-42.log"
    with artifact.registered_handle(path, registry) as handle:
        handle.write("original")
    original_stat = artifact.os.fstat
    mutated = False

    def after_stat(fd):
        nonlocal mutated
        info = original_stat(fd)
        if info.st_ino == path.stat().st_ino and not mutated:
            mutated = True
            with path.open("a") as target:
                target.write("growth")
        return info

    monkeypatch.setattr(artifact.os, "fstat", after_stat)
    manifest = artifact.stage(root, RUN, tmp_path / "out", registry=registry)
    assert manifest["output_bytes"] == 8
    entry = manifest["files"][0]
    assert entry["original_bytes"] == 8
    assert entry["observed_after_bytes"] == 14
    assert entry["changed_during_copy"] is True
    assert (tmp_path / "out/worker-42.log").read_text() == "original"


@pytest.mark.parametrize("failure", ["symlink", "foreign-invocation", "malformed", "oversize"])
def test_registry_protocol_failures_are_nonzero_without_export(tmp_path, failure):
    root = tmp_path / "root"
    run = root / RUN
    run.mkdir(parents=True)
    registry = tmp_path / "owners"
    registry.mkdir()
    source = run / "worker-42.log"
    with artifact.registered_handle(source, registry) as handle:
        handle.write("actual writer")
    receipt = registry / "worker-42.log.json"
    if failure == "symlink":
        saved = tmp_path / "saved-receipt"
        receipt.rename(saved)
        receipt.symlink_to(saved)
    elif failure == "foreign-invocation":
        value = json.loads(receipt.read_text())
        value["invocation"] = "ci-3293-2-3.12-0"
        receipt.write_text(json.dumps(value))
    elif failure == "malformed":
        receipt.write_text("not json")
    else:
        receipt.write_text("x" * 4097)
    result = cli(root, tmp_path / "out", registry)
    assert result.returncode == 2
    manifest = json.loads((tmp_path / "out/manifest.json").read_text())
    assert manifest["status"] == "rejected"
    assert manifest["registration_errors"] == 1
    assert manifest["output_bytes"] == 0
    assert not (tmp_path / "out/worker-42.log").exists()


def test_registered_run_directory_swap_stays_on_pinned_descriptor(tmp_path, monkeypatch):
    root = tmp_path / "root"
    run = root / RUN
    run.mkdir(parents=True)
    registry = tmp_path / "owners"
    registry.mkdir()
    with artifact.registered_handle(run / "worker-42.log", registry) as handle:
        handle.write("SAFE")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "worker-42.log").write_text("PRIVATE")
    real_open = artifact.os.open
    swapped = False

    def opened(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == "worker-42.log" and not swapped:
            swapped = True
            run.rename(root / "saved-run")
            run.symlink_to(outside, target_is_directory=True)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(artifact.os, "open", opened)
    manifest = artifact.stage(root, RUN, tmp_path / "out", registry=registry)
    assert manifest["status"] == "collected"
    assert (tmp_path / "out/worker-42.log").read_text() == "SAFE"


def test_missing_registry_handoff_is_error_but_no_handoff_is_healthy(tmp_path):
    root = tmp_path / "root"
    (root / RUN).mkdir(parents=True)
    result = cli(root, tmp_path / "broken", tmp_path / "lost-registry")
    assert result.returncode == 2
    assert json.loads((tmp_path / "broken/manifest.json").read_text())["status"] == "error"
    assert (
        artifact.main(
            ["--root", str(root), "--invocation", RUN, "--output", str(tmp_path / "absent")]
        )
        == 0
    )
    assert json.loads((tmp_path / "absent/manifest.json").read_text())["status"] == "missing"


@pytest.mark.parametrize(
    "failure",
    [
        "none",
        "normal",
        "missing",
        "invalid",
        "permission",
        "descriptor",
        "after-receipt",
        "closed-handle",
    ],
)
def test_r4_optional_registration_failure_preserves_existing_hard_timer(
    tmp_path, monkeypatch, failure
):
    import errno

    from tests import shard_stall_watchdog as watchdog

    run = tmp_path / "root" / RUN
    run.mkdir(parents=True)
    registry = tmp_path / "owners"
    registry.mkdir()
    monkeypatch.setattr(watchdog, "_RUN_DIR", run)
    monkeypatch.delenv(artifact.ENV_REGISTRY, raising=False)
    if failure != "none":
        handoff = registry
        if failure == "missing":
            handoff = tmp_path / "missing-registry"
        elif failure == "invalid":
            handoff = tmp_path / "not-a-directory"
            handoff.write_text("invalid registry")
        monkeypatch.setenv(artifact.ENV_REGISTRY, str(handoff))
    if failure in ("permission", "descriptor", "after-receipt"):
        code = errno.EACCES if failure == "permission" else errno.EMFILE
        original_registration = artifact.registered_handle

        def unavailable(*args, **kwargs):
            if failure == "after-receipt":
                # A receipt may have reached disk before the final write/close
                # failed. Denial must override it, never certify fallback bytes.
                with original_registration(*args, **kwargs):
                    pass
            raise OSError(code, "synthetic registration failure")

        monkeypatch.setattr(artifact, "registered_handle", unavailable)
    calls = []

    class FakeTimer:
        def dump_traceback_later(self, seconds, **kwargs):
            calls.append((seconds, kwargs))

        def cancel_dump_traceback_later(self):
            pass

    monkeypatch.setattr(watchdog, "faulthandler", FakeTimer())
    timer = watchdog._WorkerTimer(240, hard=(300, True))
    try:
        if failure == "closed-handle":
            assert timer._handle is not None
            timer._handle.close()
        timer.arm("tests/synthetic.py::test_case")
        assert not timer._disabled
        assert timer._armed
        assert len(calls) == 1
        assert calls[0][0] == 300
        assert calls[0][1] == {"file": timer._handle, "repeat": False, "exit": True}
        assert timer._path.read_text().startswith(watchdog.ARM_MARKER)
        assert artifact.ENV_REGISTRY not in os.environ
        if failure not in ("none", "normal"):
            manifest = artifact.stage(run.parent, RUN, tmp_path / "out", registry=registry)
            assert manifest["exit_code"] == 2
            assert manifest["status"] == "error"
            assert manifest["reason"] == "worker-registration-failed"
            assert manifest["output_bytes"] == 0
            assert manifest["files"] == []
            assert not list((tmp_path / "out").glob("worker-*.log"))
    finally:
        if timer._handle is not None:
            timer._handle.close()


def test_r3_wrong_source_type_is_nonzero_and_manifest_truthful(tmp_path):
    root = tmp_path / "regular-root"
    root.write_text("not a directory")
    registry = tmp_path / "owners"
    registry.mkdir()
    result = cli(root, tmp_path / "out", registry)
    assert result.returncode == 2
    manifest = json.loads((tmp_path / "out/manifest.json").read_text())
    assert manifest["status"] == "rejected"
    assert manifest["exit_code"] == 2
    assert manifest["output_bytes"] == 0
