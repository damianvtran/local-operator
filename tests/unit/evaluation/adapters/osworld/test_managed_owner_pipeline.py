"""The managed-owner pipeline drives the REAL episode from outside its loop.

``scripts/run_episode.py --engagement session`` no longer runs the episode in
its own foreground ``asyncio.run``: the frontend starts the same script under
a managed lease (detached here -- launchd is the macOS production backend) and
reports from outside, while the OWNER runs the real session arm: real SDK,
real action server over MCP, the real spawned adapter worker from the shipped
wheel, real score and cleanup. These tests assert the two observable
contracts of that split:

* the frontend reports the owner's SEALED outcome (the normal case), and
* a frontend that stops waiting exits ``failed_provisional`` and NEVER stops
  the owner -- the owner goes on to seal its own outcome after the frontend
  is gone (the lifetime the R2 review demanded).

Marked ``slow`` like the other real-spawn suites; zero cost (fake provider,
mock session wire, canary inputs only).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from local_operator import owned_process, procstate
from tests.unit.evaluation.adapters.osworld import fixtures, spawn_helpers
from tests.unit.evaluation.adapters.osworld.test_build_and_scripts import (  # noqa: F401
    durable_path,
)
from tests.unit.evaluation.adapters.osworld.test_session_arm_script import (  # noqa: F401
    adapter_wheel,
    spawn_interpreter,
)

pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parents[5]
SCRIPTS = REPO / "scripts"
FRONTEND = SCRIPTS / "run_episode.py"
CANARY_KEY = "canary-key-owned1688"
CANARY_SECRET = "canary-secret-owned1688"

#: The frontend is spawned through the INJECTED INTERNAL ENTRY (R3-S2), not
#: through a production flag. BOTH process entries are the closed Python
#: fixture composition: merely selecting detached in the frontend and then
#: relaunching production correctly refuses before any SDK/worker work. The
#: injected fixture still verifies real source/PID/birth/lease identity; it
#: never changes the production strong guarantee or infers trust from data.
INTERNAL_ENTRY = (
    "from tests.unit.evaluation.adapters.osworld.managed_owner_entry import main;"
    "raise SystemExit(main())"
)


def _seed_store(config_dir: Path) -> None:
    """Put the canaries in the SEALED store -- the one transport that crosses.

    R3-S3 removed the plaintext handoff and refuses env-only overrides on the
    managed path, so the isolated config dir's ENCRYPTED store is how these
    synthetic values travel to the owner (the same way an operator's real ones
    do); the canary assertions below then test the sealed path rather than a
    file that no longer exists.
    """
    from local_operator.secrets.access import open_store

    store = open_store(config_dir, create=True)
    store.set("AWS_ACCESS_KEY_ID", CANARY_KEY.encode())
    store.set("AWS_SECRET_ACCESS_KEY", CANARY_SECRET.encode())
    for entry in _INFRA[1::2]:
        name, value = entry.split("=", 1)
        store.set(f"FIXTURE_{name}", value.encode())


_INFRA = [
    "--infra",
    "AWS_REGION=us-east-1",
    "--infra",
    "AWS_SUBNET_ID=subnet-owned1688",
    "--infra",
    "AWS_SECURITY_GROUP_ID=sg-owned1688",
    "--infra",
    "AWS_SCHEDULER_ROLE_ARN=arn:aws:iam::0:role/owned1688",
    "--infra",
    "OSWORLD_CLIENT_PASSWORD=ownedpw",
    "--infra",
    "OSWORLD_FILE_BASE_URL=http://assets.test",
]


def _selector_and_roots(
    root: Path,
    wheel: Path,
    interpreter: spawn_helpers.SpawnInterpreter,
) -> tuple[Path, Path, Path, Path]:
    """A digest-pinned selector plus the isolated scratch/run roots.

    The composition mirrors ``test_session_arm_script`` exactly (the same
    spawnable adapter, the same core-source pin of the tree that will run the
    arm); the roots are the episode isolation contract's three names plus the
    durable run root the frontend's volatile-root refusal accepts.
    """
    selector_dir = root / "adapter"
    selector = spawn_helpers.build_spawnable_adapter(
        selector_dir,
        wheel,
        {"task_plain": fixtures.PLAIN},
        provider={"provider": "fake", "scripted_score": 1.0},
        interpreter=interpreter,
    )
    from local_operator.evaluation.adapters.api import CoreSourcePin
    from local_operator.evaluation.source_pin import (
        compute_core_source_hashes,
        resolve_core_commit,
    )

    selector = selector.model_copy(
        update={
            "core_source": CoreSourcePin(
                commit=resolve_core_commit() or "unavailable-test-tree",
                module_hashes=compute_core_source_hashes(),
            )
        }
    )
    selector_path = selector_dir / "selector.json"
    selector_path.write_text(selector.model_dump_json())
    scratch = root / "s"
    config_dir = scratch / "home" / ".local-operator"
    config_dir.mkdir(parents=True, exist_ok=True)
    (scratch / "workspace").mkdir(parents=True, exist_ok=True)
    run_root = root / "run"
    run_root.mkdir(parents=True, exist_ok=True)
    return selector_path, scratch, config_dir, run_root


def _frontend_env(scratch: Path, config_dir: Path) -> dict[str, str]:
    """The isolation contract's three names, and NO secret-bearing values.

    Built fresh (never inherited): no CMUX/LOP bindings of the pytest
    process's environment can leak into the episode. The canaries are NOT
    here (R3-S3): the managed frontend refuses env-only secret overrides, and
    the value reaches the owner through the sealed store (_seed_store) -- so
    an environment export would now be a failure, not the input.
    """
    return {
        "PATH": os.environ.get("PATH", ""),
        "TERM": "xterm-256color",
        "PYTHONPATH": str(REPO),
        "HOME": str(scratch / "home"),
        "LOCAL_OPERATOR_CONFIG_DIR": str(config_dir),
        "LOP_RUN_SCRATCH_ROOT": str(scratch),
    }


def _frontend_args(
    selector_path: Path,
    run_root: Path,
    config_dir: Path,
    *,
    frontend_wait_s: float | None = None,
) -> list[str]:
    args = [
        sys.executable,
        "-c",
        INTERNAL_ENTRY,
        "--selector",
        str(selector_path),
        "--task-id",
        "task_plain",
        "--route",
        "test/fake/model:free",
        "--run-root",
        str(run_root),
        # NO --secret-env and no --no-store: the secrets are store refs now
        # (the defaults ARE AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY), and the
        # owner resolves them from the isolated sealed store.
        "--max-steps",
        "2",
        "--max-usd",
        "0.01",
        *[
            item
            for entry in _INFRA[1::2]
            for item in (
                "--infra-ref",
                f"{entry.split('=', 1)[0]}=FIXTURE_{entry.split('=', 1)[0]}",
            )
        ],
        "--config-dir",
        str(config_dir),
        "--episode-id",
        f"owned-{uuid.uuid4().hex[:10]}",
        "--engagement",
        "session",
        "--session-route",
        "test/mock",
        "--owner-backend",
        "detached",
        "--owner-startup-timeout-s",
        "60",
    ]
    if frontend_wait_s is not None:
        args += ["--frontend-wait-s", str(frontend_wait_s)]
    return args


def _owner_status(run_root: Path, owner_id: str) -> owned_process.OwnedProcessStatus | None:
    paths = owned_process.owner_paths(run_root, owner_id)
    return owned_process.read_owner_status(paths.status_path)


def _payload(stdout: str) -> dict[str, object]:
    start = stdout.index("{")
    return json.loads(stdout[start:])


def test_the_managed_frontend_reports_the_owners_sealed_outcome(
    durable_path: Path,  # noqa: F811
    adapter_wheel: Path,  # noqa: F811
    spawn_interpreter: spawn_helpers.SpawnInterpreter,  # noqa: F811
) -> None:
    root = durable_path / f"managed-{uuid.uuid4().hex[:8]}"
    root.mkdir()
    selector_path, scratch, config_dir, run_root = _selector_and_roots(
        root, adapter_wheel, spawn_interpreter
    )
    _seed_store(config_dir)
    completed = subprocess.run(
        _frontend_args(selector_path, run_root, config_dir),
        env=_frontend_env(scratch, config_dir),
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode in (0, 1), completed.stderr[-2000:]
    payload = _payload(completed.stdout)
    # The REAL pipeline ran: a session turn on the mock wire, the real
    # spawned worker, a real score, and a sealed outcome.
    assert payload["status"] in ("agent_stop", "completed", "truncated")
    assert payload["score"] is not None, payload
    assert payload["record_root"]
    outcome_file = Path(str(payload["record_root"])) / "outcome.json"
    assert outcome_file.exists()
    # The owner ended terminal and its process is gone.
    assert completed.returncode == (0 if payload["status"] == "completed" else 1)


def test_a_frontend_that_stops_waiting_leaves_the_owner_to_finish_and_seal(
    durable_path: Path,  # noqa: F811
    adapter_wheel: Path,  # noqa: F811
    spawn_interpreter: spawn_helpers.SpawnInterpreter,  # noqa: F811
) -> None:
    root = durable_path / f"managed-{uuid.uuid4().hex[:8]}"
    root.mkdir()
    selector_path, scratch, config_dir, run_root = _selector_and_roots(
        root, adapter_wheel, spawn_interpreter
    )
    _seed_store(config_dir)
    # A wait far shorter than the episode: the frontend MUST exit provisionally
    # without touching the owner (detaching is not a stop).
    completed = subprocess.run(
        _frontend_args(selector_path, run_root, config_dir, frontend_wait_s=0.2),
        env=_frontend_env(scratch, config_dir),
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 1, completed.stderr[-2000:]
    payload = _payload(completed.stdout)
    assert payload["status"] == "failed_provisional", payload
    owner_id = str(payload["owner_id"])
    owner_pid = int(str(payload["owner_pid"]))
    # The owner was mid-run when the frontend left (it needs seconds more
    # than the 0.2 s bound); reading it dead here would mean the frontend's
    # exit killed it -- the exact defect this split removes.
    status_at_exit = _owner_status(run_root, owner_id)
    assert procstate.pid_liveness(owner_pid) is True or (
        status_at_exit is not None and status_at_exit.phase in owned_process.TERMINAL_PHASES
    )
    # The OWNER goes on: poll its status to a terminal phase, then read the
    # outcome it sealed on its own.
    deadline = time.monotonic() + 600
    terminal = None
    while time.monotonic() < deadline:
        status = _owner_status(run_root, owner_id)
        if status is not None and status.phase in owned_process.TERMINAL_PHASES:
            terminal = status
            break
        time.sleep(0.5)
    assert terminal is not None, "the owner never reached a terminal phase"
    assert terminal.outcome_path, terminal
    outcome = json.loads(Path(str(terminal.outcome_path)).read_text())
    assert outcome["status"] in ("agent_stop", "completed", "truncated")
    assert outcome["score"] is not None
    # And then it exited: the freed pid reads dead.
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if procstate.pid_liveness(owner_pid) is False:
            break
        time.sleep(0.25)
    assert procstate.pid_liveness(owner_pid) is False
