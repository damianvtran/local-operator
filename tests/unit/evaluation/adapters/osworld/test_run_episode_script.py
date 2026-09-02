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
    assert outcome["reportability_label"] == "reportable"
    assert outcome["comparability_label"] == "comparable"
    assert outcome["score"]["status"] == "scored" and outcome["score"]["binary"] == 1
    bundle = Path(outcome["bundle_root"])
    assert bundle.is_relative_to(run_root / "evidence")
    report = verify_bundle(bundle)
    assert report.valid, [issue.code for issue in report.issues]
    # The spawned worker published real frames into the parent's root.
    assert any(p.is_file() for p in (run_root / "artifacts").iterdir())
    # A clean episode retired its own descriptor: the inbox is empty.
    assert not list((run_root / "rescue").rglob("rescue.json"))
    # The secret values reached the worker over the pipe and nowhere else:
    # not the bundle, not the artifacts, not the rescue root, not stdout.
    everything = _all_bytes_under(run_root) + completed.stdout.encode() + completed.stderr.encode()
    assert CANARY_SECRET.encode() not in everything
    assert CANARY_KEY.encode() not in everything


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
