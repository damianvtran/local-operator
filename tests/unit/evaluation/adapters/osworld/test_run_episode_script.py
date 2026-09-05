"""``scripts/run_episode.py`` drives ONE real spawned episode to a sealed bundle.

This is the proof that the operator script the paid run will use actually
works end to end with nothing shared in memory: the script is run as a
subprocess with a selector for the REAL adapter wheel installed into a REAL
copied interpreter, the worker is spawned by the real supervisor, secrets
travel from the script's environment over the private pipe, and the outcome
comes back as JSON on stdout. Only the model is scripted (``--model-client
scripted-finish``) and only the provider is fake (``FakeProvider`` selected by
the digest-pinned workspace), because a paid proof must not run in CI.

Marked ``slow`` like the other real-spawn suites.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_operator.evaluation.evidence.verify import verify_bundle
from local_operator.evaluation.runner.route_ids import fold_model_id, unfold_model_id
from tests.unit.evaluation.adapters.osworld import fixtures, spawn_helpers
from tests.unit.evaluation.adapters.osworld.test_build_and_scripts import (  # noqa: F401
    durable_path,
)

pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parents[5]
SCRIPT = REPO / "scripts" / "run_episode.py"
CANARY_KEY = "AKIACANARY0000000001"
CANARY_SECRET = "canary-secret-value-9f8e7d6c5b4a"
INFRA = [
    "--infra",
    "AWS_REGION=us-east-1",
    "--infra",
    "AWS_SUBNET_ID=subnet-test",
    "--infra",
    "AWS_SECURITY_GROUP_ID=sg-test",
    "--infra",
    "AWS_SCHEDULER_ROLE_ARN=arn:aws:iam::0:role/test",
    "--infra",
    "OSWORLD_CLIENT_PASSWORD=pw",
    "--infra",
    "OSWORLD_FILE_BASE_URL=http://assets.test",
]


@pytest.fixture(scope="module")
def adapter_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return spawn_helpers.build_adapter_wheel(tmp_path_factory.mktemp("wheel"))


def _selector_file(root: Path, wheel: Path) -> Path:
    selector = spawn_helpers.build_spawnable_adapter(
        root,
        wheel,
        {"task_plain": fixtures.PLAIN},
        provider={"provider": "fake", "scripted_score": 1.0},
    )
    path = root / "selector.json"
    path.write_text(selector.model_dump_json())
    return path


def _run(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    # ``PYTHONPATH`` so the script resolves the checkout under test rather
    # than an installed harness; everything else in the environment is the
    # explicit mapping the test hands over.
    base = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "PYTHONPATH": str(REPO),
        **env,
    }
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env=base,
        cwd=str(REPO),
        check=False,
    )


def _checkout_version() -> str:
    import tomllib

    with (REPO / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def _all_bytes_under(root: Path) -> bytes:
    return b"\n".join(p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file())


def test_script_runs_one_spawned_episode_to_a_sealed_bundle(
    durable_path: Path, adapter_wheel: Path  # noqa: F811
) -> None:
    selector = _selector_file(durable_path / "adapter", adapter_wheel)
    run_root = durable_path / "run"

    completed = _run(
        [
            "--selector",
            str(selector),
            "--task-id",
            "task_plain",
            "--route",
            "test/fake/model:free",
            "--run-root",
            str(run_root),
            "--secret-env",
            "AWS_ACCESS_KEY_ID",
            "--secret-env",
            "AWS_SECRET_ACCESS_KEY",
            "--no-store",
            "--model-client",
            "scripted-finish",
            "--max-steps",
            "3",
            "--max-usd",
            "0.01",
            *INFRA,
        ],
        {"AWS_ACCESS_KEY_ID": CANARY_KEY, "AWS_SECRET_ACCESS_KEY": CANARY_SECRET},
    )

    assert completed.returncode == 0, completed.stderr[-3000:]
    outcome: dict[str, Any] = json.loads(completed.stdout)
    assert outcome["status"] == "completed", outcome
    # A scripted model client is plumbing proof, not a result: the runner's
    # own label says so, and the manifest names the client.
    assert outcome["reportability_label"] == "synthetic_model"
    assert outcome["comparability_label"] == "comparable"
    assert outcome["score"]["status"] == "scored" and outcome["score"]["binary"] == 1
    bundle = Path(outcome["bundle_root"])
    assert bundle.is_relative_to(run_root / "evidence")
    report = verify_bundle(bundle)
    assert report.valid, [issue.code for issue in report.issues]
    # The sealed route is the lossless fold of the model id, and the exact id
    # rides the manifest metadata beside it, so the route can be read back
    # without decoding.
    manifest = report.manifest
    assert manifest is not None
    assert manifest.requested_route.model_id == fold_model_id("fake/model:free")
    assert unfold_model_id(manifest.requested_route.model_id) == "fake/model:free"
    assert manifest.metadata["route_model_id"] == "fake/model:free"
    assert manifest.metadata["route_provider_id"] == "test"
    assert manifest.metadata["model_client"] == "scripted-finish"
    # No override was passed, so the bundle must NOT claim one: with the key
    # absent the effective instance type is fully determined by the hash-pinned
    # task file plus the documented default, and a stamp here would misreport
    # a default run as having used custom hardware.
    assert "aws_instance_type_override" not in manifest.metadata
    assert "aws_root_volume_size_override" not in manifest.metadata
    assert manifest.harness_version == _checkout_version()
    assert report.outcome is not None and report.outcome.reportable is False
    assert report.outcome.reportability_label == "synthetic_model"
    # The spawned worker published real frames into the parent's root.
    assert any(p.is_file() for p in (run_root / "artifacts").iterdir())
    # A clean episode retired its own descriptor: the inbox is empty, and
    # the per-episode directory sits where the documented sweep looks.
    assert (run_root / "rescue" / outcome["episode_id"]).is_dir()
    assert not list((run_root / "rescue").rglob("rescue.json"))
    # The secret values reached the worker over the pipe and nowhere else:
    # not the bundle, not the artifacts, not the rescue root, not stdout.
    everything = _all_bytes_under(run_root) + completed.stdout.encode() + completed.stderr.encode()
    assert CANARY_SECRET.encode() not in everything
    assert CANARY_KEY.encode() not in everything


def test_an_instance_type_override_is_disclosed_in_the_sealed_manifest(
    durable_path: Path, adapter_wheel: Path  # noqa: F811
) -> None:
    """A score run on non-default hardware must be disclosable from the bundle.

    Comparability is the whole point: an episode run on m5.xlarge instead of
    the release-default t3.xlarge is not directly comparable to one that used
    the default, and a reader of the bundle has no other way to learn that --
    ``ObservationPayload`` carries no metadata, so nothing the worker resolves
    reaches the bundle except through the manifest the parent seals.
    """

    selector = _selector_file(durable_path / "adapter", adapter_wheel)
    run_root = durable_path / "run"

    completed = _run(
        [
            "--selector",
            str(selector),
            "--task-id",
            "task_plain",
            "--route",
            "test/fake/model:free",
            "--run-root",
            str(run_root),
            "--secret-env",
            "AWS_ACCESS_KEY_ID",
            "--secret-env",
            "AWS_SECRET_ACCESS_KEY",
            "--no-store",
            "--model-client",
            "scripted-finish",
            "--max-steps",
            "3",
            "--max-usd",
            "0.01",
            *INFRA,
            "--infra",
            "AWS_INSTANCE_TYPE=m5.xlarge",
        ],
        {"AWS_ACCESS_KEY_ID": CANARY_KEY, "AWS_SECRET_ACCESS_KEY": CANARY_SECRET},
    )

    assert completed.returncode == 0, completed.stderr[-3000:]
    outcome: dict[str, Any] = json.loads(completed.stdout)
    assert outcome["status"] == "completed", outcome
    report = verify_bundle(Path(outcome["bundle_root"]))
    # Sealed AND valid: the manifest digest covers metadata, so a bundle that
    # verifies is one whose disclosure cannot have been edited after the fact.
    assert report.valid, [issue.code for issue in report.issues]
    assert report.manifest is not None
    assert report.manifest.metadata["aws_instance_type_override"] == "m5.xlarge"


def test_script_fails_pre_bundle_on_a_missing_secret_naming_only_the_ref(
    durable_path: Path, adapter_wheel: Path  # noqa: F811
) -> None:
    selector = _selector_file(durable_path / "adapter", adapter_wheel)
    run_root = durable_path / "run"

    completed = _run(
        [
            "--selector",
            str(selector),
            "--task-id",
            "task_plain",
            "--route",
            "test/fake-model",
            "--run-root",
            str(run_root),
            "--secret-env",
            "AWS_ACCESS_KEY_ID",
            "--secret-env",
            "AWS_SECRET_ACCESS_KEY",
            "--no-store",
            "--model-client",
            "scripted-finish",
            *INFRA,
        ],
        {"AWS_ACCESS_KEY_ID": CANARY_KEY},
    )

    assert completed.returncode == 2, completed.stderr[-3000:]
    outcome = json.loads(completed.stdout)
    assert outcome["status"] == "failed_pre_bundle"
    assert outcome["bundle_root"] is None
    assert outcome["diagnostic"] == "MissingSecret: missing secret AWS_SECRET_ACCESS_KEY"
    assert "AWS_SECRET_ACCESS_KEY" in completed.stderr
    # Nothing was allocated and nothing is left to reclaim.
    assert not list((run_root / "rescue").rglob("rescue.json"))
    assert not list((run_root / "evidence").iterdir())
    everything = _all_bytes_under(run_root) + completed.stdout.encode() + completed.stderr.encode()
    assert CANARY_KEY.encode() not in everything


@pytest.mark.parametrize("volatile", ["/tmp/lop-episode", "/private/tmp/lop-episode"])
def test_script_refuses_a_volatile_run_root(
    durable_path: Path, adapter_wheel: Path, volatile: str  # noqa: F811
) -> None:
    selector = _selector_file(durable_path / "adapter", adapter_wheel)
    completed = _run(
        [
            "--selector",
            str(selector),
            "--task-id",
            "task_plain",
            "--route",
            "test/fake-model",
            "--run-root",
            volatile,
            "--no-store",
            "--model-client",
            "scripted-finish",
        ],
        {},
    )
    assert completed.returncode == 2
    assert "OS may purge" in completed.stderr
    assert not Path(volatile).exists()


@pytest.mark.asyncio
async def test_a_dead_parent_leaves_a_descriptor_the_documented_sweep_finds(
    durable_path: Path, adapter_wheel: Path  # noqa: F811
) -> None:
    """MAJOR-2 guard: ``<run-root>/rescue`` is a real inbox for the README's sweep.

    A parent killed after ``reset_start`` leaves ``rescue.json`` behind. The
    README tells the operator to sweep ``<run-root>/rescue``; that sweep
    globs one level down, so the script's descriptor has to live at
    ``rescue/<episode>/rescue.json`` or the sweep reports ``[]`` over a live
    instance. The kill is simulated by running the runner's own
    ``_launch_and_prepare`` with the script's config and then abandoning it.
    """

    import importlib.util
    import sys as _sys

    from local_operator.evaluation.adapters.supervisor import AdapterSupervisor
    from local_operator.evaluation.runner.episode import EpisodeRunner
    from local_operator.evaluation.runner.rescue_sweep import sweep_rescue_root
    from local_operator.evaluation.runner.secrets import StaticSecretResolver

    spec_file = importlib.util.spec_from_file_location("run_episode", SCRIPT)
    assert spec_file is not None and spec_file.loader is not None
    script = importlib.util.module_from_spec(spec_file)
    _sys.modules.pop("run_episode", None)
    spec_file.loader.exec_module(script)

    selector_path = _selector_file(durable_path / "adapter", adapter_wheel)
    run_root = durable_path / "run"
    episode_id = "ep-parent-died"
    selector = script.AdapterSelector.model_validate(json.loads(selector_path.read_text()))
    config = script.build_config(
        run_root, episode_id=episode_id, max_steps=3, max_cycle_usd_micros=None
    )
    route = script._route_identity("test", "fake-model")
    spec = script.build_spec(
        episode_id=episode_id,
        selector=selector,
        task_id="task_plain",
        route=route,
        benchmark_id="osworld-v2",
        benchmark_release="osworld-v2-2026.08.08",
        secret_refs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        infra_values=script._parse_infra(INFRA[1::2], "benchmark_compute"),
        max_usd_micros=10_000,
        max_wall_ms=60_000,
        max_steps=3,
        metadata={},
    )
    resolver = StaticSecretResolver({"AWS_ACCESS_KEY_ID": "k", "AWS_SECRET_ACCESS_KEY": "s"})
    runner = EpisodeRunner(
        spec,
        config,
        selector=selector,
        model=script._ScriptedFinish(route),
        secrets=resolver,
        launch=AdapterSupervisor.launch,
    )
    # Up to and including ``prepare``: the real descriptor is on disk. Then
    # the parent "dies": the supervisor is torn down without cleanup.
    await runner._launch_and_prepare()
    await runner._emergency_teardown()

    descriptors = sorted(p.relative_to(run_root) for p in run_root.rglob("rescue.json"))
    assert descriptors == [Path("rescue") / episode_id / "rescue.json"], descriptors

    # The documented command: sweep ``<run-root>/rescue``. A FakeProvider
    # rescue worker answers ``attempted`` (it has no registry to look in), so
    # the entry must be FOUND and REPORTED incomplete, and the descriptor kept.
    entries = await sweep_rescue_root(run_root / "rescue", resolver)
    assert [(e.episode_id, e.complete) for e in entries] == [(episode_id, False)], entries
    assert (run_root / "rescue" / episode_id / "rescue.json").exists()

    # And a confirming rescue retires it -- the only way it leaves the inbox.
    async def confirming(descriptor: Any, **kwargs: Any) -> Any:
        from local_operator.evaluation.lifecycle import record_cleanup

        receipts = tuple(
            record_cleanup(
                descriptor.cleanup_plan,
                action.action_id,
                status="succeeded",
                evidence_code="instance-terminated",
                duration_ms=1,
            )
            for action in descriptor.cleanup_plan.actions
        )

        class _Aggregate:
            complete = True

        aggregate = _Aggregate()
        aggregate.receipts = receipts  # type: ignore[attr-defined]
        return aggregate

    entries = await sweep_rescue_root(run_root / "rescue", resolver, rescue=confirming)
    assert [(e.episode_id, e.complete) for e in entries] == [(episode_id, True)]
    assert not list((run_root / "rescue").rglob("rescue.json"))


def test_script_refuses_an_over_long_secret_naming_only_the_ref(
    durable_path: Path, adapter_wheel: Path  # noqa: F811
) -> None:
    """MAJOR-1 end to end: a PEM-sized value never reaches stdout, stderr or disk."""

    selector = _selector_file(durable_path / "adapter", adapter_wheel)
    run_root = durable_path / "run"
    long_value = "SECRETVALUE-" + "z" * 9000
    completed = _run(
        [
            "--selector",
            str(selector),
            "--task-id",
            "task_plain",
            "--route",
            "test/fake-model",
            "--run-root",
            str(run_root),
            "--secret-env",
            "AWS_ACCESS_KEY_ID",
            "--secret-env",
            "AWS_SECRET_ACCESS_KEY",
            "--no-store",
            "--model-client",
            "scripted-finish",
            *INFRA,
        ],
        {"AWS_ACCESS_KEY_ID": CANARY_KEY, "AWS_SECRET_ACCESS_KEY": long_value},
    )
    assert completed.returncode == 2, completed.stderr[-3000:]
    outcome = json.loads(completed.stdout)
    assert outcome["status"] == "failed_pre_bundle"
    assert outcome["diagnostic"] == "UnusableSecret: unusable secret AWS_SECRET_ACCESS_KEY"
    assert completed.stderr.strip() == outcome["diagnostic"]
    everything = _all_bytes_under(run_root) + completed.stdout.encode() + completed.stderr.encode()
    assert b"SECRETVALUE" not in everything and b"zzzz" not in everything


def test_a_root_volume_override_is_disclosed_in_the_sealed_manifest(
    durable_path: Path, adapter_wheel: Path  # noqa: F811
) -> None:
    """A run on a resized disk must be disclosable from the bundle alone.

    Comparability is the whole point. An episode on a larger root volume
    survives past the ~t+383s exhaustion wall where the guest's own snapd fills
    the default 2.2 GB of free space (NOT the x11grab recorder an earlier
    revision blamed -- no ffmpeg process was ever found), so its score is not
    comparable to a truncated default run -- and a reader has no other way to
    learn that, since ``ObservationPayload`` carries no metadata and nothing
    the worker resolves reaches the bundle except through the manifest the
    parent seals.

    Both overrides are passed together because that is what an operator
    escaping both failure modes actually runs, and it proves the two
    disclosures coexist rather than one clobbering the other.
    """

    selector = _selector_file(durable_path / "adapter", adapter_wheel)
    run_root = durable_path / "run"

    completed = _run(
        [
            "--selector",
            str(selector),
            "--task-id",
            "task_plain",
            "--route",
            "test/fake/model:free",
            "--run-root",
            str(run_root),
            "--secret-env",
            "AWS_ACCESS_KEY_ID",
            "--secret-env",
            "AWS_SECRET_ACCESS_KEY",
            "--no-store",
            "--model-client",
            "scripted-finish",
            "--max-steps",
            "3",
            "--max-usd",
            "0.01",
            *INFRA,
            "--infra",
            "AWS_ROOT_VOLUME_SIZE=120",
            "--infra",
            "AWS_INSTANCE_TYPE=m5.xlarge",
            "--infra",
            "benchmark_compute:OSWORLD_ENABLE_PROXY=false",
            "--infra",
            "benchmark_user_simulator:OSWORLD_USER_SIM_MODEL=synthetic-simulator",
        ],
        {"AWS_ACCESS_KEY_ID": CANARY_KEY, "AWS_SECRET_ACCESS_KEY": CANARY_SECRET},
    )

    assert completed.returncode == 0, completed.stderr[-3000:]
    outcome: dict[str, Any] = json.loads(completed.stdout)
    assert outcome["status"] == "completed", outcome
    report = verify_bundle(Path(outcome["bundle_root"]))
    # Sealed AND valid: the manifest digest covers metadata, so a bundle that
    # verifies is one whose disclosure cannot have been edited after the fact.
    assert report.valid, [issue.code for issue in report.issues]
    assert report.manifest is not None
    assert report.manifest.metadata["aws_root_volume_size_override"] == "120"
    assert report.manifest.metadata["aws_instance_type_override"] == "m5.xlarge"
    assert report.manifest.metadata["osworld_enable_proxy_override"] == "false"
