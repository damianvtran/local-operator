"""Which config a RELAY-ENGAGED runtime builds its approval gate from.

THE QUESTION THIS PINS (remote-onboarding design note §8's unresolved item,
slice (a)'s mandate): "does a relay-engaged runtime on a peer adopt the
machine's own config? the daemon/phone spawn path does; the relay engage path
was not exercised". The full-auto retention work claims that a session created
or engaged on a peer runs under THAT machine's tool-approval posture, so the
claim is only allowed on a measured substrate: drive the relay's own engage
path for real and read the running runtime's gate back over its control socket.

THE PATH, exactly: ``RelayServer._engage_locally`` -> ``launch.engage_runtime``
-> ``launch._spawn_runtime`` (the production spawn, ``dict(os.environ)`` plus
the birth extras) -> the child's ``process.amain`` -> ``spawn_owned_session`` ->
``ConfigManager(config_dir())`` reading ``tool_approval_mode``. Every one of
those links is the production one here; nothing is stubbed. The assertion is
therefore about the CHILD's own reading of its machine's config file, observed
from the child's own report over its own socket.

WHY THE CONFIG IS ``auto`` AND NOT ``ask``: ``ask`` is the key's default, so a
runtime that read no config at all would also report ``ask``. ``auto`` is a
value ONLY the isolated file carries, so "tool approvals: auto" on the report
is proof the machine's own config was read (and "ask" would be the failure the
pin exists to catch). One spawn, one discriminator, no unbounded waits on a
shared fleet — see the CI-calibrated registration budget the detachment rig
quotes (``_REGISTER_WAIT_S``); this file uses a longer bound because the engage
itself may spend up to its own 60 s deadline constructing the child first.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
from pathlib import Path
from typing import Any

import pytest

from local_operator.network.relay import RelayServer
from local_operator.session.runtime import registry
from tests.unit.session.runtime import test_runtime_detachment as detachment
from tests.unit.session.runtime.test_approval_authority_seam import _dial, _send

#: The child boots a real session and reads a real control socket: seconds, not
#: milliseconds, under a shared fleet — the suite's marker for real subprocesses.
pytestmark = pytest.mark.slow

_SESSION_ID = "engagegatesrc01"

#: The registration bound. The engage itself can spend its own 60 s deadline
#: constructing the child (relay.ENGAGE_DEADLINE_S) before this wait starts, so
#: the two together are the wall-clock worst case a genuinely slow host pays;
#: the wait is event-driven (the record is the signal) and only decides how long
#: "never" is.
_RECORD_WAIT_S = 180.0


def _seed(config_dir: Path, mode: str) -> None:
    """A resumable session on the mock provider, its machine config at ``mode``."""
    directory = config_dir / "sessions" / _SESSION_ID
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text(
        '{"id": "seed", "ts": 1, "type": "message", "payload": {"kind": "message", '
        '"role": "user", "content": [{"type": "text", "text": "seed"}]}}\n',
        encoding="utf-8",
    )
    (config_dir / "config.yml").write_text(
        f"values:\n  hosting: test\n  model_name: mock\n  tool_approval_mode: {mode}\n",
        encoding="utf-8",
    )


async def _wait_for_record(config_dir: Path, *, timeout: float = _RECORD_WAIT_S) -> Any:
    """The engaged session's registry record, once the child publishes it."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        for record, _state in registry.scan(config_dir):
            if getattr(record, "session_id", "") == _SESSION_ID:
                return record
        await asyncio.sleep(0.05)
    raise AssertionError(
        f"no record for {_SESSION_ID} within {timeout}s:\n"
        f"{detachment._log_text(config_dir)[-1500:]}"
    )


def _reap_group(pid: int) -> None:
    """Take the engaged child down by the pid the relay's spawn created.

    Scoped to THIS pid's group: the spawn is its own session leader
    (``start_new_session=True``), so the group holds exactly this runtime and
    its descendants, and no other session's tree can be behind this number.
    """
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)


@pytest.mark.asyncio
async def test_a_relay_engaged_runtime_adopts_the_machines_own_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pin: the relay's engage path builds the gate from THIS machine's file.

    ``auto`` in the isolated config, ``ask`` as the default the code would fall
    back to if the file were not read — so the report's own sentence is the
    verdict, read back over the production control socket with the production
    handshake.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    _seed(config_dir, "auto")
    detachment._isolate(monkeypatch, config_dir)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))

    server = RelayServer(root=config_dir)
    # ``_engage_locally`` is synchronous and runs its own ``asyncio.run`` (the
    # relay's real shape: the caller is a dispatch thread, not the event loop),
    # so it goes to a worker thread exactly as the relay calls it.
    error = await asyncio.to_thread(server._engage_locally, _SESSION_ID, cwd=str(config_dir))
    assert error == "", f"the relay engage did not reach a runtime: {error}"

    record = await _wait_for_record(config_dir)
    try:
        conn = await _dial(record, client="daemon")
        report = await _send(
            conn, None, {"op": "slash_result", "command": "approvals", "args": "", "images": []}
        )
        conn.close()
        payload = json.dumps(report)
        assert report["op"] == "result", report
        assert "tool approvals: auto" in payload, (
            "the relay-engaged runtime did not adopt the machine's own config: the "
            f"isolated file says auto and the runtime reported: {payload}"
        )
    finally:
        _reap_group(int(record.pid))
        # Drop the record a killed child leaves behind so a later scan in this
        # worker cannot read a dead owner as live.
        registry.scan(config_dir)
