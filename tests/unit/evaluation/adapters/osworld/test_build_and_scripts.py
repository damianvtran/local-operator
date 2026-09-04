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
        schema_version="1.3",
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


def _run(root: Path, pin: Path, out: Path, *extra: str) -> int:
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
            *extra,
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


# --- the attested version -------------------------------------------------
#
# ``version`` is an input to ``_release_digest``, and NOTHING downstream can
# catch a wrong one: ``adapter-release.json`` carries only the digest, so every
# digest stays internally consistent with whatever version went in. That is how
# a stale ``--version 0.1.0`` default survived the 0.1.1 bump and silently
# attested a distribution that was never built. These tests pin the two
# fail-closed guards that replaced it, and the digests themselves, so a future
# edit restoring a permissive default cannot pass with a green suite.


def test_an_explicit_version_disagreeing_with_the_adapter_is_refused(
    durable_path: Path, capsys: Any
) -> None:
    """A mismatched --version must refuse and leave NO artifact behind.

    Exit code alone is not the assertion that matters: the defect being
    guarded is a workspace that exists and looks pristine while attesting a
    version nobody built, so the absence of the output path is the property.
    """

    root, pin = _fixture_inputs(durable_path)
    out = durable_path / "workspace"
    assert _run(root, pin, out, "--version", "9.9.9") == 2
    err = capsys.readouterr().err
    # Both values, so the operator can see which one is wrong.
    assert "9.9.9" in err
    assert build._adapter_version() in err
    assert "--allow-version-mismatch" in err
    assert not out.exists()


def test_a_mismatched_version_builds_only_when_explicitly_allowed(
    durable_path: Path, capsys: Any
) -> None:
    """The override exists for re-attesting another build, and announces itself.

    Attesting a version other than this tree's is legitimate but deliberate,
    so it must be unreachable by accident: the flag is opt-in (``store_true``,
    default False), and taking it still prints what it did, because the
    resulting digest is not reproducible from this tree alone.
    """

    root, pin = _fixture_inputs(durable_path)
    out = durable_path / "workspace"
    assert _run(root, pin, out, "--version", "9.9.9", "--allow-version-mismatch") == 0
    captured = capsys.readouterr()
    assert "9.9.9" in captured.err and build._adapter_version() in captured.err
    assert out.exists()
    # The attested version reached the digest: same inputs, different version,
    # different release_digest -- which is the whole reason the guard exists.
    allowed_digest = json.loads(captured.out)["release_digest"]
    matching = durable_path / "workspace-matching"
    assert _run(root, pin, matching) == 0
    assert allowed_digest != json.loads(capsys.readouterr().out)["release_digest"]


def test_the_mismatch_override_cannot_be_set_by_accident(durable_path: Path, capsys: Any) -> None:
    """Omitting the flag must refuse even when everything else is identical.

    Guards the shape of the option rather than its effect: a ``store_true``
    silently changed to ``default=True``, or the flag being read from the
    environment, would make the override the default and re-open the defect
    while every other test here still passed.
    """

    root, pin = _fixture_inputs(durable_path)
    # Identical invocation, flag omitted -> refused, nothing written.
    assert _run(root, pin, durable_path / "w-off", "--version", "9.9.9") == 2
    assert not (durable_path / "w-off").exists()
    capsys.readouterr()
    # Same again with the flag -> allowed. The ONLY difference is the flag.
    assert (
        _run(root, pin, durable_path / "w-on", "--version", "9.9.9", "--allow-version-mismatch")
        == 0
    )
    assert (durable_path / "w-on").exists()


def test_a_version_that_cannot_be_determined_refuses_to_build(
    durable_path: Path, capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No declared version and no --version must produce no artifact.

    Reachable in practice by running the script from outside the repository
    (``parents[1]`` then misses the adapter's pyproject), which is simulated
    here by pointing the module's pyproject constant at a path that does not
    exist. A placeholder version would mint a real, well-formed workspace
    attesting something unverifiable, and ``discovery.verify_release_manifest``
    requires ``adapter-release.json`` to be exactly ``{"release_digest": ...}``
    -- so the artifact could not record its own doubt even if we wanted it to.
    """

    monkeypatch.setattr(build, "_ADAPTER_PYPROJECT", durable_path / "absent" / "pyproject.toml")
    root, pin = _fixture_inputs(durable_path)
    out = durable_path / "workspace"
    assert _run(root, pin, out) == 2
    err = capsys.readouterr().err
    assert "cannot determine the adapter version" in err
    assert not out.exists()
    # ...but an explicit version still builds: the refusal is about not
    # KNOWING, not about the file being absent.
    assert _run(root, pin, out, "--version", "0.1.1") == 0
    assert out.exists()


def test_the_staged_pilot_release_digest_is_reproduced_from_committed_values() -> None:
    """The pinned attestation of the staged pilot, through the REAL resolution.

    This is the regression test the original defect needed: it fails outright
    on the bug (the stale ``0.1.0`` version yields ``a15961b1...``) rather than
    only on a guard's error message.

    It routes through ``resolve_attested_version`` -- the same call ``main``
    makes -- rather than recomputing the rule. Calling ``_adapter_version()``
    and passing the result straight to ``_release_digest`` would re-implement
    the resolution, and a mutation to it would leave this test green: exactly
    the bypass the original defect exploited, one level up.

    Hermetic because every input is committed or supplied here -- the release
    name and task-hash manifest sha from ``config/release-v2026.08.08.json``,
    plus the package digest of the wheel the pilot installed. The COMPANION
    ``workspace_digest`` is deliberately not asserted: it hashes the 4.2 GB
    gated corpus, an operator-fetched input CI does not have and must never
    download. That half is covered structurally by the happy-path test above
    (a workspace ``workspace_digest`` accepts, rebuilt identically) and by the
    operator's build record.
    """

    pin = json.loads(build._DEFAULT_PIN.read_text())
    # Exactly what main does: nothing requested, so the tree's own declaration
    # is what gets attested.
    version, advisory = build.resolve_attested_version(
        requested=None,
        declared=build._adapter_version(),
        allow_mismatch=False,
    )
    assert advisory is None
    digest = build._release_digest(
        version=version,
        # The digest of the 0.1.1 wheel installed into the staged pilot venv.
        package_digest="69e8504d9caa6732940ec59030dc149f83549da7155fb828fa9f7de677d5a736",
        benchmark_release=pin["release"],
        task_manifest_sha256=pin["tasks"]["hash_manifest_sha256"],
    )
    assert digest == "d0067e23af3dc2ed790c2a8b802ee453200d516466ad27770d7f2bbe7b0b41cd"


def test_main_takes_its_attested_version_from_the_resolver(
    durable_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """``main`` must OBTAIN the version from ``resolve_attested_version``.

    Asserted structurally -- the resolver is replaced and its return value must
    be what reaches ``_release_digest`` -- rather than by comparing versions.
    A value comparison cannot detect this wiring at all: on the default path
    ``args.version`` is None and the resolver returns ``declared``, and on the
    override path the resolver returns ``args.version`` unchanged, so
    "resolved" and "raw" are numerically identical on every reachable input.
    Re-deriving the version at the call site is therefore an EQUIVALENT mutant
    to any value assertion (verified: it passes one), while still being the
    defect that matters -- two implementations of the rule, free to drift.

    Runs against the fixture corpus, so it needs none of the gated inputs.
    """

    root, pin = _fixture_inputs(durable_path)
    sentinel = "resolver-sentinel-version"
    calls: list[dict[str, Any]] = []
    seen: list[str] = []

    def fake_resolver(
        *, requested: str | None, declared: str | None, allow_mismatch: bool
    ) -> tuple[str, str | None]:
        calls.append(
            {"requested": requested, "declared": declared, "allow_mismatch": allow_mismatch}
        )
        return sentinel, None

    real_digest = build._release_digest

    def spy(*, version: str, **kwargs: Any) -> str:
        seen.append(version)
        return real_digest(version=version, **kwargs)

    monkeypatch.setattr(build, "resolve_attested_version", fake_resolver)
    monkeypatch.setattr(build, "_release_digest", spy)

    assert _run(root, pin, durable_path / "w") == 0
    # main asked the resolver, handing it the real declared version...
    assert calls == [
        {"requested": None, "declared": build._adapter_version(), "allow_mismatch": False}
    ]
    # ...and attested exactly what it answered, rather than re-deriving it.
    assert seen == [sentinel]
    capsys.readouterr()


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
    # ``AWS_DEFAULT_PROFILE`` too: botocore reads it alongside ``AWS_PROFILE``,
    # so omitting it leaves the fixture non-hermetic for any developer who
    # exports that spelling instead (see ``_hermetic_aws`` in
    # ``test_aws_provider.py``).
    for name in ("AWS_PROFILE", "AWS_DEFAULT_PROFILE", "AWS_DEFAULT_REGION", "AWS_REGION"):
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
