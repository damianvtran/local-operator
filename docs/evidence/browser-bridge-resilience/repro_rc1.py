"""Deterministic reproduction of RC1: a transient publish() error kills the
heartbeat task forever, so state.available() goes False permanently while the
daemon keeps serving /health 200.

Run against a checkout to see which behaviour it has:

    .venv/bin/python docs/evidence/browser-bridge-resilience/repro_rc1.py

Exit code 0 = the loop SURVIVED (fixed build). Exit code 1 = the loop DIED
(the main-branch bug). Uses a temp root, a real uvicorn server on an ephemeral
port, and a real HTTP /health probe — no mocking of the transport.
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

# Running a script puts the SCRIPT's directory on sys.path[0], not the cwd, so
# an editable install elsewhere on the machine would otherwise win the import
# and the evidence would describe the wrong checkout. Pin the repo this file
# lives in, ahead of everything else.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from local_operator.browser_bridge import state as state_store  # noqa: E402
from local_operator.browser_bridge.daemon import create_app  # noqa: E402


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


async def main() -> int:
    import uvicorn

    package = Path(__import__("local_operator").__file__ or ".").parent
    print(f"[checkout]  {package}")
    root = Path(tempfile.mkdtemp(prefix="lop-rc1-"))
    port = free_port()
    app = create_app(port, root)
    service = app.state.bridge
    # Pretend a browser is attached: availability requires extension_connected.
    # publish() recomputes extension_connected from the LINK on every tick, so
    # a fake socket has to stay set for the file to keep advertising a browser
    # — otherwise the heartbeat honestly republishes "no extension attached"
    # and availability is false for a reason that has nothing to do with RC1.
    #
    # Deliberately NOT paired: availability does not require pairing (an
    # unpaired-but-connected extension is advertised so the tool can return the
    # actionable 'lop browser pair' error). Faking paired=True without a
    # pairing FILE makes the revocation watcher correctly sever the link after
    # REVOKE_WATCH_S, which would clear extension_connected for a reason that
    # has nothing to do with the heartbeat bug under test.
    service.link.websocket = object()  # type: ignore[assignment]

    # Shrink the cadence so the reproduction takes seconds, not minutes. The
    # bug is about the loop dying, not about the interval length.
    state_store.HEARTBEAT_INTERVAL_S = 0.2

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    serving = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)

    def _health_sync() -> int:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
            return int(response.status)

    # The probe must not block the loop the server is serving from, or it
    # deadlocks against itself. Off-thread keeps it a real HTTP request.
    async def health() -> int:
        return await asyncio.to_thread(_health_sync)

    def heartbeat_age() -> float:
        raw = json.loads((root / "run/browser/bridge.json").read_text())
        return time.time() - float(raw["heartbeat_at"])

    await asyncio.sleep(0.6)
    print(
        f"[baseline]  /health={await health()}  available={state_store.available(root)}  "
        f"heartbeat_age={heartbeat_age():.2f}s"
    )

    # Inject the exact failure the machine hit: ENOSPC out of publish().
    real_publish = state_store.publish
    failures = {"count": 0}

    def exploding_publish(state, root_arg=None):  # type: ignore[no-untyped-def]
        failures["count"] += 1
        if failures["count"] <= 5:
            raise OSError(28, "No space left on device")
        return real_publish(state, root_arg)

    state_store.publish = exploding_publish  # type: ignore[assignment]
    print("[inject]    publish() now raises OSError(ENOSPC) for 5 calls")
    await asyncio.sleep(2.0)
    state_store.publish = real_publish  # type: ignore[assignment]
    print("[recover]   publish() restored; disk is 'free' again")
    await asyncio.sleep(1.0)

    task = service._heartbeat_task
    task_dead = task is None or task.done()
    age = heartbeat_age()
    status = await health()
    live = state_store.available(root)
    current = state_store.read(root)
    # Print every term of the availability predicate, not just its verdict:
    # a bare False is indistinguishable between "heartbeat stale" and "the
    # test rig never marked an extension attached".
    print(
        f"[after]     /health={status}  available={live}  heartbeat_age={age:.2f}s  "
        f"heartbeat_task_dead={task_dead}"
    )
    if current is not None:
        print(
            f"[terms]     extension_connected={current.extension_connected}  "
            f"pid_alive={state_store.pid_alive(current.pid)}  "
            f"liveness={state_store.liveness(root)[0].value}"
        )

    # The decisive consequence. A dead writer never refreshes the file again,
    # so its age CLIMBS in real time and crosses HEARTBEAT_TIMEOUT_S for good:
    # unavailable not just now but permanently. A live writer holds the age
    # near zero. Sampling the age twice measures exactly that, without having
    # to sit out the full 45s timeout.
    #
    # (Evaluating available() at a simulated future timestamp would be
    # meaningless here: it asks "would this file be stale in 46s" while a
    # healthy daemon is about to rewrite it 230 times in that window.)
    first_age = heartbeat_age()
    await asyncio.sleep(1.0)
    second_age = heartbeat_age()
    climbing = second_age > first_age + 0.5
    print(
        f"[verdict]   age {first_age:.2f}s -> {second_age:.2f}s  "
        f"writer_stopped={climbing}  available_now={state_store.available(root)}"
    )

    server.should_exit = True
    await serving

    if task_dead or not live or climbing:
        print(
            "\nRESULT: BROKEN — heartbeat writer died on a transient error; the daemon "
            "answers /health 200 but every session now sees the bridge as unavailable."
        )
        return 1
    print(
        "\nRESULT: SURVIVED — the heartbeat loop absorbed the errors, kept looping, and "
        "republished a fresh heartbeat once the write succeeded again."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
