"""Build drift is an announcement, not permission to cancel daemon-owned work.

Use production timing and the actual ``serve --port 0`` command. A marker-only
fake cannot supply a successor, and no stream pins these isolated daemons open.
"""

import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

OLD = "1" * 40
NEW = "2" * 40
NEWER = "3" * 40


def wait_for(predicate, timeout=25):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError("isolated daemon condition did not become true")


@pytest.mark.parametrize("scheduler", [False, True], ids=["production-cli", "scheduler-barrier"])
def test_build_drift_keeps_production_daemon_serving(tmp_path: Path, scheduler: bool):
    home, root, prefix = (
        tmp_path / name for name in ("serve-home", "serve-config", "serve-prefix")
    )
    for directory in (home, root, prefix):
        directory.mkdir()
    marker = prefix / ".lop-source"
    marker.write_text(f"{OLD} old-build\n")
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("CMUX_", "LOP_"))
        and key not in {"LOCAL_OPERATOR_DESKTOP_TOKEN", "LOCAL_OPERATOR_DESKTOP_ORIGINS"}
    }
    token = secrets.token_hex(32)
    env.update(
        HOME=str(home),
        LOCAL_OPERATOR_CONFIG_DIR=str(root),
        LOCAL_OPERATOR_DESKTOP_TOKEN=token,
        LOP_BUILD_PREFIX=str(prefix),
    )
    command = (
        [sys.executable, "-m", "tests.e2e.serve_scheduler_barrier"]
        if scheduler
        else [str(Path(sys.executable).parent / "local-operator")]
    ) + ["serve", "--host", "127.0.0.1", "--port", "0"]
    with (tmp_path / "daemon.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        record_path = root / "run" / "serve" / f"{process.pid}.json"

        def record():
            return json.loads(record_path.read_text()) if record_path.exists() else {}

        try:
            wait_for(record_path.exists)
            initial = record()
            url = f"http://127.0.0.1:{initial['port']}"
            with httpx.Client(base_url=url, headers={"Authorization": f"Bearer {token}"}) as client:
                wait_for(lambda: client.get("/health").status_code == 200)
                if scheduler:
                    wait_for(lambda: (root / "task-started").exists())
                unauthorized = client.post(
                    "/v1/desktop/sessions",
                    headers={"Authorization": ""},
                    json={"cwd": str(home), "request_id": str(uuid4())},
                )
                assert unauthorized.status_code == 401
                invalid = client.post("/v1/desktop/sessions", json={"cwd": str(home)})
                assert invalid.status_code == 422
                print("unauthorized mutation=401; missing request_id=422", flush=True)
                before_heartbeat = initial["heartbeat_at"]
                marker.write_text(f"{NEW} new-build\n")
                started = time.monotonic()
                statuses = set()
                # 50 s exceeds settle + two checks + the maximum stagger (40 s).
                # This deliberately catches the production default, not just a
                # test callback or shortened poll that could mask assembly bugs.
                while time.monotonic() - started < 50:
                    if process.poll() is not None:
                        break
                    try:
                        statuses.add(client.get("/health").status_code)
                        statuses.add(
                            client.post(
                                "/v1/desktop/sessions",
                                json={"cwd": str(home), "request_id": str(uuid4())},
                            ).status_code
                        )
                    except httpx.TransportError:
                        break
                    time.sleep(1)
                elapsed = time.monotonic() - started
                print(
                    f"scheduler={scheduler} elapsed={elapsed:.1f}s "
                    f"same_pid={process.poll() is None} HTTP={sorted(statuses)} "
                    f"task_cancelled={(root / 'task-cancelled').exists()}",
                    flush=True,
                )
                assert process.poll() is None, "build watcher exited the production daemon"
                assert elapsed >= 50, "HTTP transport failed before the survival window ended"
                assert statuses == {200}, "build watcher refused new work"
                current = record()
                assert current["pid"] == initial["pid"]
                assert current["heartbeat_at"] > before_heartbeat
                assert current["retiring_to"].endswith(NEW[:7])
                assert current["claim_key"] == initial["claim_key"]
                print(
                    "heartbeat advanced; announcement readable; claim key unchanged=True",
                    flush=True,
                )
                if scheduler:
                    assert not (root / "task-cancelled").exists()
                    assert not (root / "task-completed").exists()
                    (root / "task-release").touch()
                    wait_for(lambda: (root / "task-completed").exists())
                    assert (root / "task-completed").read_text() == "completed\n"
                    print("barrier released; exactly one completed side effect", flush=True)
                marker.write_text(f"{OLD} old-build\n")
                wait_for(lambda: not record().get("retiring_to"))
                assert (
                    client.post(
                        "/v1/desktop/sessions", json={"cwd": str(home), "request_id": str(uuid4())}
                    ).status_code
                    == 200
                )
                marker.write_text(f"{NEW} new-build\n")
                wait_for(lambda: record().get("retiring_to", "").endswith(NEW[:7]))
                marker.write_text(f"{NEWER} newer-build\n")
                wait_for(lambda: record().get("retiring_to", "").endswith(NEWER[:7]))
                assert (
                    client.post(
                        "/v1/desktop/sessions", json={"cwd": str(home), "request_id": str(uuid4())}
                    ).status_code
                    == 200
                )
                assert not (root / "task-cancelled").exists()
                print(
                    "withdrawal and retarget verified; later authenticated mutation=200", flush=True
                )
        finally:
            # Test-owned process only; this teardown is explicitly separate from
            # the build watcher and occurs after the survival/completion evidence.
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=20)
