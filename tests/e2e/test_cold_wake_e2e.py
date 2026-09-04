"""A wake for a CLOSED session runs its turn with no terminal open.

The supervisor's entire purpose, and this release's headline promise, is that
a scheduled wake fires even when the session is not open. Round 2 (U4/Q9)
found the cold path did everything except the last step: the runtime started,
advanced the schedule, and exited having made zero provider calls — a one-shot
wake permanently lost without ever running.

These tests drive the same actors the production path uses — `wake create`'s
persist shape, an overdue schedule the supervisor would fire, and the runtime
child's boot — and assert the turn landed in the durable transcript.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e


def _seed(config_dir: Path, session_id: str) -> None:
    directory = config_dir / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.yml").parent.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text(
        json.dumps(
            {
                "id": "seed",
                "ts": 1,
                "type": "message",
                "payload": {
                    "kind": "message",
                    "role": "user",
                    "content": [{"type": "text", "text": "seed"}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (config_dir / "config.yml").write_text(
        "values:\n  hosting: test\n  model_name: mock\n", encoding="utf-8"
    )


def test_wake_create_persists_through_the_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`wake create` writes the source of truth, not only the derived index.

    An index-only entry is adopted as nothing and deleted by the next open —
    the session rebuilds its index from the transcript on every open. That
    made the created wake invisible to the very runtime the supervisor
    started for it.
    """
    from local_operator.cli import _wake_create
    from local_operator.harness.wake import WAKE_SCHEDULES_CUSTOM_TYPE
    from local_operator.session.transcript import Transcript

    # `_wake_create` resolves the store through `paths.config_dir()`, which
    # reads the environment on every call — the same seam the unit tests use.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _seed(tmp_path, "wakecreate01")
    args = argparse.Namespace(
        session="wakecreate01", when="in 2m", message="check the build", json=False
    )
    assert _wake_create(args) == 0

    transcript = Transcript(tmp_path / "sessions" / "wakecreate01")
    entry = transcript.latest_custom_entry(WAKE_SCHEDULES_CUSTOM_TYPE)
    assert entry is not None, "the schedule never reached the source of truth"
    schedules = entry.payload["details"]["schedules"]
    assert schedules and schedules[0]["message"] == "check the build"


def test_a_cold_wake_runs_its_turn_with_no_terminal_open(tmp_path: Path) -> None:
    """The round-2 blocker, end to end.

    The runtime child must arm the scheduler at boot (``async_init``) and the
    catch-up must be delivered at the grace deadline even when nothing else
    ever happens in the session — a cold runtime has no prompt head to
    trigger it. Asserted on the durable transcript: the wake delivery row AND
    the assistant reply it spawned.
    """
    from local_operator.harness.wake import WAKE_SCHEDULES_CUSTOM_TYPE
    from local_operator.session.transcript import Transcript
    from local_operator.wakes.store import write_entry

    _seed(tmp_path, "coldwake0001")
    now = int(time.time() * 1000)
    # OVERDUE, as a wake the supervisor fires for a session that was closed.
    schedule = {
        "id": "w1",
        "message": "cold wake says hello",
        "next_due_at": now - 5_000,
        "created_at": now - 60_000,
    }
    transcript = Transcript(tmp_path / "sessions" / "coldwake0001")
    asyncio.run(transcript.append_custom(WAKE_SCHEDULES_CUSTOM_TYPE, {"schedules": [schedule]}))
    write_entry(tmp_path, "coldwake0001", cwd=str(tmp_path), schedules=[schedule])

    env = {
        **os.environ,
        "LOCAL_OPERATOR_CONFIG_DIR": str(tmp_path),
        "LOP_MOBILE_CHILD_CWD": str(tmp_path),
        "LOP_MOBILE_CHILD_RESUME": "coldwake0001",
    }
    child = subprocess.Popen(
        [sys.executable, "-m", "local_operator.session.runtime.process"],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        transcript_path = tmp_path / "sessions" / "coldwake0001" / "transcript.jsonl"
        deadline = time.monotonic() + 45
        text = ""
        while time.monotonic() < deadline:
            text = transcript_path.read_text(encoding="utf-8")
            if "Hello from the mock provider!" in text and "wake_prompt" in text:
                break
            time.sleep(0.5)
        else:
            raise AssertionError(f"the cold wake never ran its turn; transcript:\n{text}")
    finally:
        child.kill()
        child.wait(timeout=10)
