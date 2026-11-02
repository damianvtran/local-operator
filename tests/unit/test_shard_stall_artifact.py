"""Synthetic-only retention boundary controls; never boot a native worker."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests import shard_stall_artifact as artifact
from tests import shard_stall_watchdog as watchdog

RUN = "ci-123-2-3.12-0"


@pytest.fixture(autouse=True)
def isolated_handoff(monkeypatch):
    # install() exports the run directory to workers; restore it between controls.
    monkeypatch.delenv(watchdog.ENV_RUN_DIR, raising=False)
    monkeypatch.delenv(artifact.ENV_INVOCATION, raising=False)
    monkeypatch.delenv(artifact.ENV_REGISTRY, raising=False)


def owned_file(path, payload):
    registry = path.parent.parent.parent / "owners"
    registry.mkdir(exist_ok=True)
    with artifact.registered_handle(path, registry) as handle:
        handle.buffer.write(payload)


def collect(tmp_path, payload=b"[shard stall] armed\n", **limits):
    root = tmp_path / "lo-shard-stall"
    run = root / RUN
    run.mkdir(parents=True)
    owned_file(run / "worker-42.log", payload)
    out = tmp_path / "artifact"
    manifest = artifact.stage(root, RUN, out, registry=tmp_path / "owners", **limits)
    return root, run, out, manifest


def test_all_snapshots_preserved_without_modifying_source(tmp_path):
    data = b"[shard stall] armed\nTimeout (0:04:00)!\nfirst\nTimeout (0:04:00)!\nlater\n"
    root, run, out, manifest = collect(tmp_path, data)
    assert (out / "worker-42.log").read_bytes() == data
    assert (run / "worker-42.log").read_bytes() == data
    assert manifest["files"][0]["status"] == "retained"
    assert manifest["files"][0]["timeout_marker"] == "present"
    assert manifest["files"][0]["original_bytes"] == len(data)
    assert manifest["invocation"] == RUN
    assert manifest["provenance"] == {"run": "123", "attempt": "2", "python": "3.12", "shard": "0"}
    assert json.loads((out / "manifest.json").read_text()) == manifest


def test_header_only_is_not_called_a_timeout(tmp_path):
    _, _, _, manifest = collect(tmp_path)
    assert manifest["files"][0]["timeout_marker"] == "absent-in-retained-prefix"


def test_exact_invocation_excludes_stale_and_neighbor_files(tmp_path):
    root, _, out, _ = collect(tmp_path)
    for name in ("run-999", "ci-123-1-3.12-0", "ci-123-2-3.12-1"):
        (root / name).mkdir()
        (root / name / "worker-999.log").write_text("stale")
    manifest = artifact.stage(root, RUN, tmp_path / "second", registry=tmp_path / "owners")
    assert [f["name"] for f in manifest["files"]] == ["worker-42.log"]
    assert not (out / "worker-999.log").exists()


@pytest.mark.parametrize("kind", ["root", "run", "worker", "fifo"])
def test_symlinks_and_nonregular_inputs_rejected(tmp_path, kind):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    if kind == "root":
        root = tmp_path / "linked-root"
        root.symlink_to(outside, target_is_directory=True)
    elif kind == "run":
        (root / RUN).symlink_to(outside, target_is_directory=True)
    else:
        (root / RUN).mkdir()
        target = root / RUN / "worker-1.log"
        owned_file(target, b"original writer")
        target.unlink()
        if kind == "worker":
            secret = outside / "secret"
            secret.write_text("must not export")
            target.symlink_to(secret)
        else:
            os.mkfifo(target)
    manifest = artifact.stage(root, RUN, tmp_path / "artifact", registry=tmp_path / "owners")
    assert "rejected" in [manifest["status"], *[f["status"] for f in manifest["files"]]]
    assert not list((tmp_path / "artifact").glob("worker-*.log"))


def test_missing_is_explicit_and_healthy(tmp_path):
    manifest = artifact.stage(tmp_path / "missing", RUN, tmp_path / "artifact")
    assert manifest["status"] == "missing"
    assert manifest["files"] == []


@pytest.mark.parametrize("data", [b"abc\x00secret", b"\xff\xfe"])
def test_nontext_not_exported(tmp_path, data):
    _, _, out, manifest = collect(tmp_path, data)
    assert manifest["files"][0]["status"] == "rejected"
    assert manifest["files"][0]["output_bytes"] == 0
    assert not (out / "worker-42.log").exists()


def test_file_total_and_count_caps_are_truthful(tmp_path):
    root, run, _, _ = collect(tmp_path, b"a" * 20)
    owned_file(run / "worker-43.log", b"b" * 20)
    owned_file(run / "worker-44.log", b"c" * 20)
    manifest = artifact.stage(
        root,
        RUN,
        tmp_path / "capped",
        file_cap=8,
        total_cap=10,
        count_cap=2,
        registry=tmp_path / "owners",
    )
    assert [f["status"] for f in manifest["files"]] == ["truncated", "truncated", "omitted-cap"]
    assert [f["output_bytes"] for f in manifest["files"]] == [8, 2, 0]
    assert [f["original_bytes"] for f in manifest["files"]] == [20, 20, 20]
    assert manifest["output_bytes"] == 10
    assert (run / "worker-42.log").stat().st_size == 20


@pytest.mark.parametrize(
    "run_id", ["../outside", "/tmp/outside", "run-42", "ci-1-2-3.12-0/../x", ""]
)
def test_identity_protocol_fails_closed(tmp_path, run_id):
    with pytest.raises(ValueError, match="invocation"):
        artifact.stage(tmp_path, run_id, tmp_path / "artifact")
    assert not (tmp_path / "artifact").exists()


def test_missing_empty_run_is_explicit(tmp_path):
    root = tmp_path / "root"
    (root / RUN).mkdir(parents=True)
    (tmp_path / "owners").mkdir()
    manifest = artifact.stage(root, RUN, tmp_path / "artifact", registry=tmp_path / "owners")
    assert manifest["status"] == "missing"
    assert manifest["reason"] == "no-current-invocation-worker-files"


def test_hardlinks_and_arbitrary_scratch_are_not_exported(tmp_path):
    root, run, _, _ = collect(tmp_path)
    (run / "worker-43.log").hardlink_to(run / "worker-42.log")
    (run / "fixture-private.txt").write_text("private scratch")
    manifest = artifact.stage(root, RUN, tmp_path / "second", registry=tmp_path / "owners")
    assert len(manifest["files"]) == 1
    assert all(f["status"] == "rejected" for f in manifest["files"])
    assert not list((tmp_path / "second").glob("*.log"))


def test_directory_scan_and_manifest_are_bounded(tmp_path):
    root, run, _, _ = collect(tmp_path)
    for index in range(artifact.SCAN_CAP + 5):
        owned_file(run / f"worker-{1000 + index}.log", b"stack")
    manifest = artifact.stage(root, RUN, tmp_path / "second", registry=tmp_path / "owners")
    assert len(manifest["files"]) <= artifact.SCAN_CAP
    assert manifest["directory_tail"]["status"] == "omitted-cap"
    assert manifest["directory_tail"]["count"] is None
    assert len(list((tmp_path / "second").glob("*.log"))) == artifact.COUNT_CAP


def test_utf8_cap_does_not_invent_encoding_damage(tmp_path):
    _, _, out, manifest = collect(tmp_path, "a\u20acb".encode(), file_cap=3)
    assert (out / "worker-42.log").read_text() == "a"
    assert manifest["files"][0]["status"] == "truncated"
    assert manifest["files"][0]["output_bytes"] == 1


def test_total_cap_independent_of_count_cap(tmp_path):
    root, run, _, _ = collect(tmp_path, b"abc")
    owned_file(run / "worker-43.log", b"xyz")
    manifest = artifact.stage(
        root, RUN, tmp_path / "second", total_cap=3, registry=tmp_path / "owners"
    )
    assert [f["status"] for f in manifest["files"]] == ["retained", "omitted-cap"]
    assert manifest["output_bytes"] == 3


def test_source_replacement_cannot_follow_symlink(tmp_path, monkeypatch):
    root, run, _, _ = collect(tmp_path)
    secret = tmp_path / "secret"
    secret.write_text("not a worker stack")
    original_open = artifact.os.open

    def swapped_open(path, flags, *args, **kwargs):
        if path == "worker-42.log":
            (run / path).unlink()
            (run / path).symlink_to(secret)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(artifact.os, "open", swapped_open)
    manifest = artifact.stage(root, RUN, tmp_path / "second", registry=tmp_path / "owners")
    assert manifest["files"][0]["status"] == "rejected"
    assert not (tmp_path / "second/worker-42.log").exists()


def test_source_disappearing_during_collection_is_missing(tmp_path, monkeypatch):
    root, run, _, _ = collect(tmp_path)
    original_open = artifact.os.open

    def vanished_open(path, flags, *args, **kwargs):
        if path == "worker-42.log":
            (run / path).unlink()
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(artifact.os, "open", vanished_open)
    manifest = artifact.stage(root, RUN, tmp_path / "second", registry=tmp_path / "owners")
    assert manifest["files"][0]["status"] == "missing"


def test_cli_protocol_failure_does_not_export(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.shard_stall_artifact",
            "--root",
            str(tmp_path),
            "--invocation",
            "../outside",
            "--output",
            str(tmp_path / "artifact"),
        ],
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "staging refused: ValueError" in result.stderr
    assert not (tmp_path / "artifact").exists()


def test_controller_rejects_path_protocol_without_failing_pytest(tmp_path, monkeypatch):
    monkeypatch.setenv(artifact.ENV_INVOCATION, "../outside")
    monkeypatch.setattr(watchdog, "DUMP_ROOT", tmp_path / "root")
    monkeypatch.setattr(watchdog, "_RUN_DIR", None)
    monkeypatch.setattr(watchdog, "_CONTROLLER", None)
    monkeypatch.setattr(watchdog._Controller, "start", lambda self: None)
    monkeypatch.setenv(watchdog.ENV_SECONDS, "240")
    watchdog.install(object(), on_ci=True)
    assert watchdog.run_dir() == tmp_path / "root" / f"run-{os.getpid()}"
    assert not (tmp_path / "outside").exists()


def test_controller_identity_and_real_cli_workflow_contract(tmp_path, monkeypatch):
    monkeypatch.setenv(artifact.ENV_INVOCATION, RUN)
    monkeypatch.setattr(watchdog, "DUMP_ROOT", tmp_path / "lo-shard-stall")
    monkeypatch.setattr(watchdog, "_RUN_DIR", None)
    monkeypatch.setattr(watchdog, "_CONTROLLER", None)
    monkeypatch.setattr(watchdog._Controller, "start", lambda self: None)
    monkeypatch.setenv(watchdog.ENV_SECONDS, "240")
    watchdog.install(object(), on_ci=True)
    assert watchdog.run_dir() == tmp_path / "lo-shard-stall" / RUN
    assert artifact.ENV_INVOCATION not in os.environ
    # A child controller starts with inherited environment, not inherited module
    # state. Simulate that boundary without launching any worker or process.
    monkeypatch.setattr(watchdog, "_RUN_DIR", None)
    monkeypatch.setattr(watchdog, "_CONTROLLER", None)
    watchdog.install(object(), on_ci=True)
    assert watchdog.run_dir().name == f"run-{os.getpid()}"
    monkeypatch.setattr(watchdog, "_RUN_DIR", tmp_path / "lo-shard-stall" / RUN)
    owned_file(watchdog.run_dir() / "worker-1.log", b"Timeout (0:04:00)!\nfirst\nlater\n")
    workflow = yaml.safe_load((Path(__file__).parents[2] / ".github/workflows/ci.yml").read_text())
    steps = workflow["jobs"]["test"]["steps"]
    stage = next(s for s in steps if s.get("id") == "stage_stall")
    pytest_step = next(s for s in steps if s.get("id") == "shard_pytest")
    assert stage["env"][artifact.ENV_INVOCATION] == pytest_step["env"][artifact.ENV_INVOCATION]
    assert stage["env"][artifact.ENV_REGISTRY] == "${{ steps.shard_pytest.outputs.stall_registry }}"
    assert pytest_step["run"].index("export " + artifact.ENV_REGISTRY) < pytest_step["run"].index(
        "pytest $("
    )
    upload = next(s for s in steps if s.get("name") == "Upload current-invocation timeout files")
    assert upload["with"]["retention-days"] == 1
    assert upload["with"]["path"] == "${{ steps.stage_stall.outputs.artifact }}"
    command = stage["run"].replace(
        '"${TMPDIR:-/tmp}/lo-shard-stall"', '"' + str(tmp_path / "lo-shard-stall") + '"'
    )
    outputs = tmp_path / "github-outputs"
    result = subprocess.run(
        ["bash", "-eu", "-c", command],
        cwd=Path(__file__).parents[2],
        env={
            **os.environ,
            artifact.ENV_INVOCATION: RUN,
            artifact.ENV_REGISTRY: str(tmp_path / "owners"),
            "GITHUB_OUTPUT": str(outputs),
            "RUNNER_TEMP": str(tmp_path),
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        },
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    output = Path(outputs.read_text().strip().removeprefix("artifact="))
    assert output.parent.name.startswith("lo-stall-stage.")
    assert json.loads((output / "manifest.json").read_text())["files"][0]["status"] == "retained"
