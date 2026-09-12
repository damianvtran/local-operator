"""A runtime recovers a session whose claim is held by a zombie, and runs.

The incident, end to end and with every actor real: an operator's session whose
runtime was killed while its parent — a long-lived TUI — lived on, so the
transcript's sole-writer claim named a pid that had exited but never been
reaped. `kill(pid, 0)` succeeds against such a pid, and every caller that asked
only that question concluded the session was busy: `lop sessions` showed it as
merely "stored" (discovery DOES probe for zombies, so it reaped the record)
while the TUI's `/resume`, `lop exec --resume` and the phone all refused with
"already open in another process (pid N)", N being the corpse. Neither
`acquire_session_lease` nor `reap_proven_dead_session_claim` would touch the
claim, because both require a holder to be PROVEN dead.

This test drives the production child (`python -m
local_operator.session.runtime.process`, the same argv ``launch._spawn_runtime``
uses) against a session directory planted with exactly that state, and asserts
on the durable transcript that the turn ran. Before the fix the child logged
"runtime lost the lease for ... to pid <corpse>; exiting" and no turn ever
landed — so the failure here is at the last step and for the right reason.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.unreaped import unreaped_child

pytestmark = pytest.mark.e2e

SESSION_ID = "zombieleas01"


def _seed(config_dir: Path, session_id: str) -> None:
    """A resumable session on the mock provider, as the cold-wake stage seeds it."""
    directory = config_dir / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
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


def test_a_runtime_takes_over_a_claim_held_by_a_zombie(tmp_path: Path) -> None:
    from local_operator.harness.wake import WAKE_SCHEDULES_CUSTOM_TYPE
    from local_operator.session.transcript import Transcript
    from local_operator.wakes.store import write_entry

    _seed(tmp_path, SESSION_ID)
    now = int(time.time() * 1000)
    schedule = {
        "id": "w1",
        "message": "hello after the zombie",
        "next_due_at": now - 5_000,
        "created_at": now - 60_000,
    }
    transcript = Transcript(tmp_path / "sessions" / SESSION_ID)
    asyncio.run(transcript.append_custom(WAKE_SCHEDULES_CUSTOM_TYPE, {"schedules": [schedule]}))
    write_entry(tmp_path, SESSION_ID, cwd=str(tmp_path), schedules=[schedule])

    session_dir = tmp_path / "sessions" / SESSION_ID
    # HOME as well as the config dir: the cache root is derived from the home
    # directory independently, so a run isolated by config dir alone still
    # reads and writes the operator's real cache (see AGENTS.md).
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "LOCAL_OPERATOR_CONFIG_DIR": str(tmp_path),
        "LOP_MOBILE_CHILD_CWD": str(tmp_path),
        "LOP_MOBILE_CHILD_RESUME": SESSION_ID,
    }

    with unreaped_child() as zombie_pid:
        claim = session_dir / ".execution-lease"
        child_log = tmp_path / "runtime-child.log"
        claim.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "session_id": SESSION_ID,
                    "generation": "c" * 32,
                    "pid": zombie_pid,
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        (session_dir / ".session.pid").write_text(str(zombie_pid), encoding="utf-8")

        child = subprocess.Popen(
            [sys.executable, "-m", "local_operator.session.runtime.process"],
            env=env,
            stdin=subprocess.DEVNULL,
            # A FILE, not a pipe: nothing drains a pipe here, and 64 KB of
            # runtime logging would block the child mid-boot.
            stdout=child_log.open("wb"),
            stderr=subprocess.STDOUT,
        )
        try:
            transcript_path = session_dir / "transcript.jsonl"
            deadline = time.monotonic() + 45
            text = ""
            while time.monotonic() < deadline:
                text = transcript_path.read_text(encoding="utf-8")
                if "Hello from the mock provider!" in text and "wake_prompt" in text:
                    break
                if child.poll() is not None:
                    break
                time.sleep(0.5)
            else:
                raise AssertionError(f"the recovery turn never ran; transcript:\n{text}")
            if child.poll() is not None and "Hello from the mock provider!" not in text:
                raise AssertionError(
                    f"the runtime child exited {child.returncode} without taking the "
                    f"claim over; transcript:\n{text}\n"
                    f"child output:\n{child_log.read_text(errors='replace')[-4000:]}"
                )
        finally:
            child.kill()
            child.wait(timeout=10)

        # The claim that outlives this test is the runtime's, not the corpse's:
        # the child rewrote it when it took over, so a later attach is refused by
        # the live owner rather than recovering a dead one twice.
        recovered = json.loads(claim.read_text(encoding="utf-8"))
        assert recovered["pid"] != zombie_pid
        assert recovered["generation"] != "c" * 32
