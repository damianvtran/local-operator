"""A move must never delete a session a record-LESS process is writing.

The recorded case (a runtime that published a discovery record) is refused by the
runtime's own ``exclusive`` fence and pinned by ``test_mobility``/``_crash``. THIS
file pins the case that fence cannot see: a live process holding the transcript
lease while publishing NO runtime record — ``lop exec``, the headless REPL, the
server, or a runtime mid-boot. ``find_runtime_record`` answers ``(None, pid)`` for
it, and the source's retire used to read that as ``cold`` and let the commit
delete the directory under the writer.

The holder here is a REAL separate process that takes the lease through
``acquire_session_lease`` exactly as those callers do, so the probe under test
reads a genuine claim (pid + birth token), not a hand-written file.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from local_operator.network import sync
from local_operator.network.projection import read_tombstones
from local_operator.session.placement import read_handoff_journal
from local_operator.session_lease import LEASE_NAME, lease_holder
from tests.unit.network.test_mobility import (  # noqa: F401 — fixtures
    SESSION,
    Devices,
    _move,
    _owned_session,
    _transcript,
    devices,
    pair,
)
from tests.unit.network.test_relay_e2e import _pair

_HOLDER = """
import sys, time
from pathlib import Path
from local_operator.session_lease import acquire_session_lease
lease = acquire_session_lease(Path(sys.argv[1]))
print("held", flush=True)
time.sleep(120)
"""


def _hold_lease(session_dir: Path) -> subprocess.Popen[str]:
    """Start a process that holds ``session_dir``'s lease and publishes nothing."""
    child = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(session_dir)],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert child.stdout is not None
    assert child.stdout.readline().strip() == "held", "the holder never took the lease"
    return child


def _reap(child: subprocess.Popen[str]) -> None:
    """Kill exactly the holder we started, by its own process group."""
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGKILL)
    child.wait(timeout=10)


def test_the_probe_reads_a_live_claim_without_taking_it(tmp_path: Path) -> None:
    directory = tmp_path / "sessions" / SESSION
    directory.mkdir(parents=True)
    assert lease_holder(directory) == (None, "none")
    child = _hold_lease(directory)
    try:
        before = (directory / LEASE_NAME).read_bytes()
        assert lease_holder(directory) == (child.pid, "live")
        # NOT ACQUIRED: the claim on disk is still the holder's, byte for byte.
        assert (directory / LEASE_NAME).read_bytes() == before
    finally:
        _reap(child)
    assert lease_holder(directory) == (None, "none")


def test_a_move_of_a_session_a_recordless_process_holds_is_refused_then_succeeds(
    pair: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811 — the imported fixture
) -> None:
    """Refused with nothing mutated while held; moves once the lease is released."""
    server_a, server_b, _host, _port = pair
    _pair(pair, monkeypatch, role="admin")
    source = _owned_session(server_a)
    before = _transcript(server_a.root, SESSION)

    child = _hold_lease(source)
    try:
        refused = _move(server_b, SESSION, monkeypatch=monkeypatch)
        assert refused["ok"] is False, refused
        assert refused["code"] == "busy", refused
        assert f"pid {child.pid}" in refused["message"], refused
        assert "published no runtime record" in refused["message"], refused
        assert refused["changed"] is False
        # NOTHING CHANGED ON EITHER SIDE: the writer's directory, its bytes and
        # its claim are intact, and B holds neither a session nor a staging copy.
        assert source.is_dir() and _transcript(server_a.root, SESSION) == before
        assert child.poll() is None, "the move killed the holder"
        assert lease_holder(source) == (child.pid, "live")
        assert read_handoff_journal(server_a.root) == {}
        assert read_tombstones(server_a.root) == {}
        assert not (server_b.root / "sessions" / SESSION).exists()
        assert not sync.staging_dir(server_b.root, SESSION).exists()
    finally:
        _reap(child)

    # RELEASED (the holder is gone and its claim proven dead): the same move lands.
    deadline = time.monotonic() + 10
    while lease_holder(source)[1] != "none" and time.monotonic() < deadline:
        time.sleep(0.05)
    moved = _move(server_b, SESSION, monkeypatch=monkeypatch)
    assert moved["ok"] is True, moved
    assert not source.exists()
    assert _transcript(server_b.root, SESSION) == before
