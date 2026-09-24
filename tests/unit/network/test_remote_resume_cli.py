"""``lop --resume <a peer's id>`` from a SHELL, over a real pair of relays.

WHY THIS IS A SUBPROCESS TEST, and not a unit cell: the pre-check this pins lives
inside ``cli.main`` and its whole subject is what a user SEES at a shell prompt —
one line on stdout, or the existing refusal on stderr plus the non-zero status.
Driving the real entry point is the only way to assert that, and the cost is one
interpreter per case.

WHAT IT PINS (mesh slice DB2, from the cross-host matrix's finding): a peer's id
used to reach the LOCAL resolver first, so a conversation the sidebar lists
perfectly well answered "no session … to resume" from a shell while the TUI's own
``/resume`` opened it. The two cases are the two sentences:

* an id ANOTHER DEVICE holds ⇒ "``<id>`` lives on ``<device>`` — opening it
  remotely", and the local refusal is NOT printed;
* an id nobody holds ⇒ the SAME refusal as before this change, unchanged.

THE THIRD CASE IS DELIBERATELY NOT HERE: a peer that has become UNREACHABLE does
not appear in this device's listing at all (a peer that does not answer
contributes a peer block and no rows — ``relay._fan_out_catalog``), so a shell
that has never listed the mesh falls through to the ordinary refusal rather than
to the unreachable sentence. The unreachable sentence is reached from a CACHED
row, which is the TUI's case and is pinned in the server and TUI suites.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from tests.unit.network.test_relay_e2e import devices  # noqa: F401 — fixtures
from tests.unit.network.test_session_plane import (
    Devices,
    _create_named_session_on_a_real_peer,
)

#: How long a CLI run gets. Generous on purpose: the child imports the whole
#: application (seconds under fleet load), reads the store and — in the peer's
#: case — goes on to launch a TUI, which is the run this test INTERRUPTS. The
#: assertion is on the line the pre-check prints BEFORE any of that.
RUN_TIMEOUT_S = 60.0


@pytest.fixture()
def peer_pair(request: pytest.FixtureRequest) -> Devices:
    """The two real relays ``test_relay_e2e`` builds, under this file's own name.

    Named here rather than imported from the sibling suite: a fixture is that
    file's interface to its own tests, and importing one module's test into another
    is how a rename in one breaks the other for no reason a reader can see.
    """
    pair: Devices = request.getfixturevalue("devices")
    return pair


def _run_cli(root: Path, home: Path, *args: str) -> tuple[int | None, str, str]:
    """Run ``lop`` as a user would, in its own process group, and reap it.

    THE ENVIRONMENT IS BUILT FROM SCRATCH rather than inherited: ``env -i``'s
    equivalent for a subprocess. ``LOP_*`` decides what a child runtime IS
    (``session/runtime/process.py``) and ``CMUX_*`` names the operator's real
    workspaces, so an inherited value would make this cell measure the session
    that started it — the failure ``AGENTS.md`` records twice.
    """
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(home),
        "LOCAL_OPERATOR_CONFIG_DIR": str(root),
        "TERM": "xterm-256color",
        "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "local_operator.cli", *args],
        env=env,
        cwd=str(home),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        out, err = process.communicate(timeout=RUN_TIMEOUT_S)
        return process.returncode, out, err
    except subprocess.TimeoutExpired:
        # THE PEER'S CASE REACHES A TUI, and a TUI on a pipe is not this test's
        # subject: the pre-check's line is already written, so the run is cut short
        # here and the output so far is what gets asserted. The whole GROUP is
        # killed (a TUI is a process tree, not a process).
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        out, err = process.communicate()
        return None, out or "", err or ""


@pytest.mark.asyncio
async def test_a_peers_id_resumes_from_a_shell_and_an_unknown_one_still_refuses(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    created = await asyncio.to_thread(
        _create_named_session_on_a_real_peer,
        peer_pair,
        monkeypatch,
        name="shell-resume",
        prompt="",
    )
    try:
        home = tmp_path / "shell-home"
        home.mkdir()
        root = created.server_a.root

        # THE UNKNOWN ID: unchanged, and asserted FIRST so a regression in the new
        # arm cannot mask it (this cell runs the refusal the change was not allowed
        # to touch).
        code, out, err = await asyncio.to_thread(_run_cli, root, home, "--resume", "feedfacecafe")
        assert code == 1, (code, out, err)
        assert "no session 'feedfacecafe' to resume" in err, err
        assert "lives on" not in out, out

        # THE PEER'S ID: said where it lives, and NOT refused.
        code, out, err = await asyncio.to_thread(
            _run_cli, root, home, "--resume", created.session_id
        )
        line = next((row for row in out.splitlines() if "lives on" in row), "")
        assert line.startswith(f"{created.session_id} lives on "), (code, out, err)
        assert line.endswith("— opening it remotely"), (code, out, err)
        # AND THE LOCAL REFUSAL IS NOT PRINTED: that sentence is the defect this
        # arm exists to remove, and it would otherwise arrive from the resolver one
        # line later.
        assert "to resume" not in err, (
            "a conversation another device holds was answered as a typo: " + err
        )
    finally:
        await asyncio.to_thread(created.stop)
