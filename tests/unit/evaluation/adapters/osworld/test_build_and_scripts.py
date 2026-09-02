"""The operator scripts: workspace build, tag audit, rescue sweep.

The build script is the only path that materialises a production workspace,
so its failure mode -- REFUSE on any hash mismatch, exit 4 with the path --
is asserted against a fixture inputs root whose every pin is derived from the
fixture bytes, then tampered one byte at a time. The happy path must produce
a workspace ``discovery.workspace_digest`` accepts (no links, all files
read-only).

The audit and sweep are asserted against stubbed clients and a fake
``run_rescue``: the property under test is what each does with the answer
(prints ``[]``/exit 0, unlinks only on ``complete``), not AWS itself.
"""

from __future__ import annotations

import hashlib
import json
import stat
import subprocess
from pathlib import Path
from typing import Any

import boto3
import pytest
from botocore.stub import ANY, Stubber
from lop_osworld_v2_adapter.providers.aws import _Clients

from local_operator.evaluation.adapters.discovery import workspace_digest
from local_operator.evaluation.adapters.supervisor import persist_rescue
from local_operator.evaluation.lifecycle import CleanupReceipt, record_cleanup
from scripts import build_osworld_adapter as build
from scripts import osworld_rescue_sweep as sweep
from scripts import osworld_tag_audit as audit
from tests.unit.evaluation.adapters.osworld import fixtures
from tests.unit.evaluation.runner.conftest import cleanup_plan, handshake, selector

RELEASE = "osworld-v2-2026.08.08"
COMMIT = "d578d2d4e0dc82b43e270fdaa7fa89d9708cd154"


def _real_home() -> Path:
    """The account's real home, not the scratch HOME the suite's conftest sets.

    ``pwd`` rather than ``Path.home()`` because the root conftest re-points
    ``HOME`` at a per-test scratch directory under pytest's basetemp -- which
    is exactly the volatile location these tests must avoid.
    """

    import os
    import pwd

    return Path(pwd.getpwuid(os.getuid()).pw_dir)


@pytest.fixture
def durable_path(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A scratch directory the build script's volatile-root refusal accepts.

    pytest's ``tmp_path`` is ``$TMPDIR``-derived on macOS but a literal
    ``/tmp/pytest-of-<user>/...`` on Linux runners, and the build script now
    refuses BOTH as an inputs root (round-1 MINOR-1 -- a purge of /tmp
    destroyed a paid pilot's inputs). Re-pointing ``$TMPDIR`` only fixes the
    macOS case, which is exactly how CI went red while the local run stayed
    green. A root under the repo is no better: this repo is worked through
    worktrees that themselves live under ``/tmp``. So the build tests get a
    root under the REAL home directory -- durable on every host that can run
    the suite -- created per test and removed afterwards. The refusal itself
    is never patched: the two literal-``/tmp`` tests and the ``$TMPDIR`` test
    run against the unmodified function.
    """

    import shutil
    import uuid

    root = _real_home() / ".cache" / "lop-osworld-build-tests" / uuid.uuid4().hex[:12]
    root.mkdir(parents=True)
    # ``$TMPDIR`` is re-pointed under the durable root too, so a host whose
    # ``$TMPDIR`` is an ancestor of home (a container might do it) cannot
    # turn the home path volatile.
    (root / "tmpdir").mkdir()
    monkeypatch.setenv("TMPDIR", str(root / "tmpdir"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def runner_descriptor(tmp_path: Path, episode_id: str, *, secret_refs: Any = ()) -> Any:
    from local_operator.evaluation.adapters.api import RescueDescriptor

    tmp_path.mkdir(parents=True, exist_ok=True)
    return RescueDescriptor(
        schema_version="1.2",
        selector=selector(tmp_path),
        handshake=handshake(tmp_path),
        episode_id=episode_id,
        cleanup_plan=cleanup_plan(episode_id),
        secret_refs=secret_refs,
        infra_values=(),
        artifact_root=str(tmp_path),
    )


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture_inputs(tmp_path: Path, *, task_count: int = 3) -> tuple[Path, Path]:
    """A fixture inputs root plus a release pin whose hashes match it."""

    root = tmp_path / "inputs"
    prepared = root / "prepared"
    gated = root / "gated"
    (prepared / "benchmark_releases").mkdir(parents=True)
    (gated / "tasks" / "manifests").mkdir(parents=True)
    (gated / "manifests").mkdir(parents=True)

    release_manifest = json.dumps({"release": RELEASE}).encode()
    (prepared / "benchmark_releases" / f"{RELEASE}.json").write_bytes(release_manifest)

    tasks: dict[str, bytes] = {}
    for index in range(task_count):
        name = f"task_{index + 1:03d}.py"
        data = fixtures.PLAIN.replace("task_plain", f"task_{index + 1:03d}").encode()
        (gated / "tasks" / name).write_bytes(data)
        tasks[name] = data
    hash_manifest = json.dumps(
        {
            "files": {
                name: {"sha256": _sha(data), "size": len(data)} for name, data in tasks.items()
            },
            "task_count": task_count,
        },
        sort_keys=True,
    ).encode()
    (gated / "tasks" / "manifests" / "task_hashes.json").write_bytes(hash_manifest)
    (gated / "manifests" / "assets.json").write_bytes(json.dumps({"revision": "acad110e"}).encode())

    # A real git checkout at a known commit is expensive; the script shells
    # out to ``git rev-parse HEAD``, so a minimal repo with one commit gives
    # a genuine (if arbitrary) HEAD the pin then names.
    subprocess.run(["git", "init", "-q", str(prepared)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(prepared),
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "pin",
        ],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(prepared), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    pin = {
        "schema_version": 1,
        "release": RELEASE,
        "release_manifest_sha256": _sha(release_manifest),
        "osworld": {"repository": "x", "tag": "t", "commit": head},
        "tasks": {
            "repository": "x",
            "tag": "t",
            "revision": "r",
            "hash_manifest_path": "manifests/task_hashes.json",
            "hash_manifest_sha256": _sha(hash_manifest),
            "task_count": task_count,
        },
        "assets": {"repository": "x", "tag": "t", "revision": "acad110e"},
    }
    pin_path = tmp_path / "pin.json"
    pin_path.write_text(json.dumps(pin))
    return root, pin_path


def _run(root: Path, pin: Path, out: Path) -> int:
    return build.main(
        [
            "--benchmark-release",
            RELEASE,
            "--out",
            str(out),
            "--inputs-root",
            str(root),
            "--release-pin",
            str(pin),
        ]
    )


def test_happy_path_builds_a_digestable_readonly_workspace(durable_path: Path, capsys: Any) -> None:
    root, pin = _fixture_inputs(durable_path)
    out = durable_path / "workspace"
    assert _run(root, pin, out) == 0
    digest = workspace_digest(str(out))
    assert len(digest) == 64
    names = sorted(p.name for p in out.iterdir())
    assert names == [
        "adapter-provider.json",
        "adapter-release.json",
        "benchmark_release.json",
        "inputs.json",
        "task_hashes.json",
        "tasks",
    ]
    assert sorted(p.name for p in (out / "tasks").iterdir()) == [
        "task_001.py",
        "task_002.py",
        "task_003.py",
    ]
    for path in out.rglob("*"):
        if path.is_file():
            assert not path.is_symlink()
            assert not (path.stat().st_mode & stat.S_IWUSR), path
    assert json.loads((out / "adapter-provider.json").read_text()) == {"provider": "aws"}
    inputs = json.loads((out / "inputs.json").read_text())
    assert inputs["assets_manifest_sha256"] == _sha(
        (root / "gated" / "manifests" / "assets.json").read_bytes()
    )
    assert inputs["tasks_manifest_sha256"] == _sha(
        (root / "gated" / "tasks" / "manifests" / "task_hashes.json").read_bytes()
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["task_count"] == 3
    # Rebuilding over the read-only workspace succeeds (the operator re-runs it).
    assert _run(root, pin, out) == 0
    assert workspace_digest(str(out)) == digest


@pytest.mark.parametrize(
    ("relative", "mutate", "message"),
    [
        ("gated/tasks/task_002.py", lambda b: b + b"#", "task file sha256 mismatch"),
        ("gated/tasks/manifests/task_hashes.json", lambda b: b + b" ", "task hash manifest sha256"),
        (
            "prepared/benchmark_releases/" + RELEASE + ".json",
            lambda b: b + b" ",
            "release manifest sha256",
        ),
        (
            "gated/manifests/assets.json",
            lambda b: b'{"revision": "other"}',
            "assets manifest revision",
        ),
    ],
)
def test_any_tampered_input_fails_closed_with_exit_4(
    durable_path: Path, capsys: Any, relative: str, mutate: Any, message: str
) -> None:
    root, pin = _fixture_inputs(durable_path)
    target = root / relative
    target.write_bytes(mutate(target.read_bytes()))
    out = durable_path / "workspace"
    assert _run(root, pin, out) == build.EXIT_VERIFY
    err = capsys.readouterr().err
    assert message in err
    assert relative.rsplit("/", 1)[-1] in err
    assert not out.exists()


def test_a_missing_task_and_a_wrong_count_both_fail(durable_path: Path, capsys: Any) -> None:
    root, pin = _fixture_inputs(durable_path)
    (root / "gated" / "tasks" / "task_003.py").unlink()
    assert _run(root, pin, durable_path / "w1") == build.EXIT_VERIFY
    assert "task file missing" in capsys.readouterr().err

    root, pin = _fixture_inputs(durable_path / "second")
    payload = json.loads(pin.read_text())
    payload["tasks"]["task_count"] = 108
    pin.write_text(json.dumps(payload))
    assert _run(root, pin, durable_path / "w2") == build.EXIT_VERIFY
    assert "pin says 108" in capsys.readouterr().err


@pytest.mark.parametrize("volatile", ["/tmp/osworld-inputs", "/private/tmp/osworld-inputs"])
def test_an_inputs_root_under_tmp_is_refused_before_verification(
    durable_path: Path, capsys: Any, volatile: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The documented constraint is enforced: a purge of /tmp destroyed a paid
    pilot's inputs mid-run. Refused BEFORE any manifest is read, so a root that
    does not even exist under /tmp still fails on the location, not on a
    missing file."""

    _root, pin = _fixture_inputs(durable_path)
    out = durable_path / "workspace"
    assert (
        build.main(
            [
                "--benchmark-release",
                RELEASE,
                "--out",
                str(out),
                "--inputs-root",
                volatile,
                "--release-pin",
                str(pin),
            ]
        )
        == build.EXIT_VERIFY
    )
    err = capsys.readouterr().err
    assert "OS may purge" in err
    assert not out.exists()


def test_a_root_under_tmpdir_is_refused_too(
    durable_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TMPDIR", str(durable_path / "scratch"))
    (durable_path / "scratch").mkdir()
    with pytest.raises(build.VerificationFailed, match="OS may purge"):
        build.refuse_volatile_root(durable_path / "scratch" / "inputs")
    # A sibling of the volatile root is fine.
    build.refuse_volatile_root(durable_path / "elsewhere")


def test_a_wrong_prepared_commit_fails(durable_path: Path, capsys: Any) -> None:
    root, pin = _fixture_inputs(durable_path)
    payload = json.loads(pin.read_text())
    payload["osworld"]["commit"] = COMMIT
    pin.write_text(json.dumps(payload))
    assert _run(root, pin, durable_path / "w") == build.EXIT_VERIFY
    assert "prepared checkout HEAD" in capsys.readouterr().err


def test_the_committed_release_pin_carries_the_known_hashes() -> None:
    pin = json.loads(build._DEFAULT_PIN.read_text())
    assert pin["release"] == RELEASE
    assert pin["osworld"]["commit"] == COMMIT
    assert pin["tasks"]["task_count"] == 108
    assert pin["tasks"]["hash_manifest_sha256"] == (
        "42f8f6f8939b8712997d5891456a575f8a2a5f53465e9e3e6747af5d6efd0915"
    )
    assert pin["release_manifest_sha256"] == (
        "afe4f61ba6f4e4dce6c9f5815578e41e084fb6b61ee96b7118d9055e5d339aab"
    )


# ---------------------------------------------------------------------------
# tag audit
# ---------------------------------------------------------------------------


@pytest.fixture
def stubbed_clients(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    for name in ("AWS_PROFILE", "AWS_DEFAULT_REGION", "AWS_REGION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-config"))
    ec2 = boto3.client(
        "ec2", region_name="us-east-1", aws_access_key_id="x", aws_secret_access_key="y"
    )
    scheduler = boto3.client(
        "scheduler", region_name="us-east-1", aws_access_key_id="x", aws_secret_access_key="y"
    )
    with Stubber(ec2) as ec2_stub, Stubber(scheduler) as sched_stub:
        yield _Clients(ec2=ec2, scheduler=scheduler, http_get=lambda u, t: 0), ec2_stub, sched_stub


def test_audit_prints_empty_list_and_exits_zero_when_clean(
    stubbed_clients: Any, capsys: Any
) -> None:
    clients, ec2_stub, sched_stub = stubbed_clients
    ec2_stub.add_response("describe_instances", {"Reservations": []}, {"Filters": ANY})
    ec2_stub.add_response("describe_volumes", {"Volumes": []}, {"Filters": ANY})
    sched_stub.add_response("list_schedules", {"Schedules": []}, {"NamePrefix": "lop-ttl-"})
    assert audit.main(["--region", "us-east-1"], clients=clients) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_audit_exits_one_and_lists_a_leaked_instance(stubbed_clients: Any, capsys: Any) -> None:
    clients, ec2_stub, sched_stub = stubbed_clients
    ec2_stub.add_response(
        "describe_instances",
        {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": "i-leak",
                            "State": {"Name": "running"},
                            "Tags": [{"Key": "lop:episode", "Value": "ep-x"}],
                        }
                    ]
                }
            ]
        },
        {"Filters": ANY},
    )
    ec2_stub.add_response("describe_volumes", {"Volumes": []}, {"Filters": ANY})
    sched_stub.add_response("list_schedules", {"Schedules": []}, {"NamePrefix": "lop-ttl-"})
    assert audit.main(["--region", "us-east-1"], clients=clients) == 1
    found = json.loads(capsys.readouterr().out)
    assert [(f["kind"], f["id"], f["episode"]) for f in found] == [("instance", "i-leak", "ep-x")]


# ---------------------------------------------------------------------------
# rescue sweep
# ---------------------------------------------------------------------------


class _Aggregate:
    def __init__(self, complete: bool, receipts: tuple[CleanupReceipt, ...]) -> None:
        self.complete = complete
        self.receipts = receipts


@pytest.mark.asyncio
async def test_sweep_unlinks_only_descriptors_whose_rescue_completed(tmp_path: Path) -> None:
    root = tmp_path / "rescue"
    complete = runner_descriptor(tmp_path / "a", "ep-complete")
    stuck = runner_descriptor(tmp_path / "b", "ep-stuck")
    persist_rescue(root / "ep-complete", complete)
    persist_rescue(root / "ep-stuck", stuck)
    # A directory without a descriptor is skipped, not an error.
    (root / "ep-empty").mkdir()

    seen: list[tuple[str, tuple[str, ...]]] = []

    async def fake_rescue(descriptor: Any, *, secrets: Any) -> _Aggregate:
        seen.append((descriptor.episode_id, tuple(s.name for s in secrets)))
        action = descriptor.cleanup_plan.actions[0]
        if descriptor.episode_id == "ep-complete":
            receipt = record_cleanup(
                descriptor.cleanup_plan,
                action.action_id,
                status="succeeded",
                evidence_code="instance-terminated",
                duration_ms=1,
            )
            return _Aggregate(True, (receipt,))
        receipt = record_cleanup(
            descriptor.cleanup_plan,
            action.action_id,
            status="attempted",
            evidence_code="terminate-unconfirmed",
            duration_ms=1,
        )
        return _Aggregate(False, (receipt,))

    class _Resolver:
        def resolve(self, names: Any) -> Any:
            from local_operator.evaluation.adapters.api import ResolvedSecret

            return tuple(ResolvedSecret(name=n, value="v") for n in names)

    entries = await sweep.sweep_rescue_root(root, _Resolver(), rescue=fake_rescue)
    assert [(e.episode_id, e.complete, e.codes) for e in entries] == [
        ("ep-complete", True, ("instance-terminated",)),
        ("ep-stuck", False, ("terminate-unconfirmed",)),
    ]
    assert not (root / "ep-complete" / "rescue.json").exists()
    assert (root / "ep-stuck" / "rescue.json").exists()
    assert [episode for episode, _ in seen] == ["ep-complete", "ep-stuck"]


@pytest.mark.asyncio
async def test_sweep_reports_a_missing_secret_and_keeps_the_descriptor(tmp_path: Path) -> None:
    from local_operator.evaluation.adapters.api import SecretRef

    root = tmp_path / "rescue"
    with_ref = runner_descriptor(
        tmp_path / "a", "ep-needs-key", secret_refs=(SecretRef(name="AWS_SECRET_ACCESS_KEY"),)
    )
    persist_rescue(root / "ep-needs-key", with_ref)

    from local_operator.evaluation.runner.secrets import MissingSecret

    class _Resolver:
        def resolve(self, names: Any) -> Any:
            raise MissingSecret(names[0])

    async def never(descriptor: Any, *, secrets: Any) -> Any:  # pragma: no cover
        raise AssertionError("rescue must not run without its secrets")

    entries = await sweep.sweep_rescue_root(root, _Resolver(), rescue=never)
    assert len(entries) == 1
    assert entries[0].complete is False
    assert entries[0].error == "missing secret AWS_SECRET_ACCESS_KEY"
    assert (root / "ep-needs-key" / "rescue.json").exists()
