"""Registration follows the successful detached spawn, not leader liveness.

The real cell is producer-only: its cooperative child exits via an owned gate
file (with self-expiry), never a signal/reaper. A delayed scheduler can hand the
already-reaped shell back to execute_bash while that child's group is live.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shlex
import sys
import time
from types import SimpleNamespace

import pytest

from local_operator import memory_guard, procstate
from local_operator.harness.types import ToolContext
from local_operator.tools import builtin, group_reaper


@pytest.fixture(autouse=True)
def no_process_signals(monkeypatch):
    attempts = []
    for name in ("kill", "killpg"):
        original = getattr(os, name, None)
        if original is None:
            continue

        def guarded(pid, sig, *, _original=original):
            if sig:
                attempts.append(sig)
                raise AssertionError("nonzero signal forbidden in producer regression")
            return _original(pid, sig)

        monkeypatch.setattr(os, name, guarded)
    yield attempts
    assert attempts == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX detached group invariant")
@pytest.mark.parametrize(
    "options,expected",
    [
        ({"start_new_session": True}, True),
        ({"creationflags": 8}, False),
        ({}, False),
    ],
)
@pytest.mark.asyncio
async def test_known_group_uses_actual_spawn_options_not_late_lookup(
    tmp_path, monkeypatch, options, expected
):
    registered, settled, guarded, spawned = [], [], [], []
    stdout, stderr = asyncio.StreamReader(), asyncio.StreamReader()
    stdout.feed_eof()
    stderr.feed_eof()

    async def wait():
        return 0

    process = SimpleNamespace(pid=54321, returncode=0, stdout=stdout, stderr=stderr, wait=wait)

    async def spawn(*args, **kwargs):
        spawned.append(kwargs)
        return process

    def late_lookup(*args):
        raise ProcessLookupError("unit leader already reaped")

    async def sample():
        return None

    def guard(pgid, budget):
        guarded.append(pgid)
        return SimpleNamespace(
            tick_s=0.25, sample=sample, should_kill=lambda _: False, soft_notice=lambda _: None
        )

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(builtin, "detached_popen_kwargs", lambda: options.copy())
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(os, "getpgid", late_lookup)
    monkeypatch.setattr(
        group_reaper, "register_group", lambda pgid, *a, **kw: registered.append(pgid)
    )
    monkeypatch.setattr(group_reaper, "settle_group", lambda pgid, *a, **kw: settled.append(pgid))
    monkeypatch.setattr(memory_guard, "Guard", guard)
    monkeypatch.setattr(
        memory_guard,
        "compute_budget",
        lambda **kwargs: memory_guard.Budget(64, 48, None, None, None, "override", "unit"),
    )
    result = await builtin.execute_bash(
        "unit",
        {"command": "printf unit", "memory_mb": 64},
        None,
        None,
        ToolContext(cwd=str(tmp_path), session_id="producer-unit"),
    )
    assert not result.is_error
    assert "exit code: 0" in json.dumps(result.model_dump(mode="json"))
    assert all(spawned[0].get(key) == value for key, value in options.items())
    wanted = [process.pid] if expected else []
    assert registered == wanted
    assert settled == wanted
    assert guarded == wanted, "memory guard must use the same proven detached group"


CHILD = """import json,os,time
from pathlib import Path
root=Path(__file__).parent
deadline=time.monotonic()+15
(root/'ready.json').write_text(json.dumps({'pid':os.getpid(),'pgid':os.getpgid(0)}))
try:
 while (root/'keep-running').exists() and time.monotonic()<deadline: time.sleep(.02)
finally:
 reason='gate_removed' if not (root/'keep-running').exists() else 'self_expiry'
 (root/'done.json').write_text(json.dumps({'reason':reason}))
"""


@pytest.mark.skipif(os.name != "posix", reason="POSIX kernel group identity")
@pytest.mark.asyncio
async def test_real_short_leader_registers_live_cooperative_survivor(
    tmp_path, monkeypatch, no_process_signals
):
    """Force the real shell.wait-before-producer ordering; never fabricate IDs."""
    config = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    gate, ready, done = (tmp_path / name for name in ("keep-running", "ready.json", "done.json"))
    gate.touch()
    script = tmp_path / "cooperative.py"
    script.write_text(CHILD)
    command = (
        f"{shlex.quote(sys.executable)} {shlex.quote(str(script))} >/dev/null 2>&1 & echo pid=$!"
    )
    real_spawn = asyncio.create_subprocess_exec
    leader, registrations, reap_attempts = [], [], []
    child_pid = None

    async def spawn(*args, **kwargs):
        assert kwargs.get("start_new_session") is True
        process = await real_spawn(*args, **kwargs)
        leader.append(process.pid)
        # A scheduler is allowed to resume the parent this late. Waiting for
        # the real Process, rather than faking getpgid's exception, proves it.
        assert await process.wait() == 0
        with pytest.raises(ProcessLookupError):
            os.getpgid(process.pid)
        return process

    original_register = group_reaper.register_group

    def register(*args, **kwargs):
        registrations.append(args[0])
        return original_register(*args, **kwargs)

    def never_reap(*args, **kwargs):
        reap_attempts.append(True)
        raise AssertionError("producer regression must not call a reaper")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(group_reaper, "register_group", register)
    monkeypatch.setattr(group_reaper, "reap_session_groups", never_reap)
    monkeypatch.setattr(group_reaper, "sweep_orphan_groups", never_reap)
    try:
        result = await builtin.execute_bash(
            "cooperative",
            {"command": command, "timeout": 20, "memory_mb": 0},
            None,
            None,
            ToolContext(cwd=str(tmp_path), session_id="producer-cooperative"),
        )
        response = json.dumps(result.model_dump(mode="json"))
        assert not result.is_error and "exit code: 0" in response
        match = re.search(r"pid=(\d+)", response)
        assert match, "setup: real child PID was not returned"
        child_pid = int(match.group(1))
        deadline = time.monotonic() + 3
        while not ready.exists() and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        assert ready.exists(), "setup: cooperative child did not report identity"
        identity = json.loads(ready.read_text())
        before = procstate.process_sample(child_pid)
        assert before is not None and before.birth is not None and not before.zombie
        assert identity == {"pid": child_pid, "pgid": leader[0]}
        assert os.getpgid(child_pid) == leader[0]
        assert not done.exists(), "setup: child expired before registration observation"
        entries = [
            entry for path in config.rglob("*.jsonl") for entry in group_reaper._read_ledger(path)
        ]
        after = procstate.process_sample(child_pid)
        assert after is not None and after.birth == before.birth and not after.zombie
        assert registrations == leader, "reaped leader skipped registration of its live group"
        assert len(entries) == 1
        entry = entries[0]
        assert entry["pgid"] == leader[0] and entry["runtime_pid"] == os.getpid()
        assert entry["runtime_start"] is not None and entry["session_id"] == "producer-cooperative"
        survivors = entry["survivors"]
        assert isinstance(survivors, list)
        assert all(isinstance(member, dict) and member.get("start") for member in survivors)
        assert child_pid in [member["pid"] for member in survivors if isinstance(member, dict)]
        assert "detached" not in entry, "do not weaken the intentional-detach classifier"
        print(
            "PRODUCER_PROOF "
            + json.dumps(
                {
                    "owner_pid": os.getpid(),
                    "leader_pid": leader[0],
                    "child_pid": child_pid,
                    "child_birth": before.birth,
                    "child_pgid": os.getpgid(child_pid),
                    "registrations": registrations,
                    "survivors": survivors,
                    "session_id": entry["session_id"],
                    "nonzero_signals": len(no_process_signals),
                }
            ),
            flush=True,
        )
    finally:
        gate.unlink(missing_ok=True)
        deadline = time.monotonic() + 17
        while time.monotonic() < deadline:
            if child_pid is None and ready.exists():
                child_pid = json.loads(ready.read_text())["pid"]
            if child_pid is not None:
                sample = procstate.process_sample(child_pid)
                if sample is not None and sample.zombie:
                    # A CI process may already be a subreaper. Reap only this
                    # verified child, without signalling it or another process.
                    with contextlib.suppress(ChildProcessError):
                        os.waitpid(child_pid, os.WNOHANG)
                    sample = procstate.process_sample(child_pid)
                if done.exists() and sample is None:
                    assert json.loads(done.read_text())["reason"] == "gate_removed"
                    print(
                        "PRODUCER_CLEANUP "
                        + json.dumps(
                            {
                                "child_pid": child_pid,
                                "child_absent": sample is None,
                                "reason": "gate_removed",
                                "nonzero_signals": len(no_process_signals),
                            }
                        ),
                        flush=True,
                    )
                    break
            await asyncio.sleep(0.02)
        else:
            pytest.fail(
                "setup cleanup: cooperative child exit not confirmed; signals remain prohibited"
            )
        assert reap_attempts == []
