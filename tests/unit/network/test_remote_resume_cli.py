"""``lop --resume <a peer's id>`` from a SHELL, over a real pair of relays.

WHY THIS IS A SUBPROCESS TEST, and not a unit cell: the pre-check this pins lives
inside ``cli.main`` and its whole subject is what a user SEES at a shell prompt —
one line on stdout, or the existing refusal on stderr plus the non-zero status.
Driving the real entry point is the only way to assert that, and the cost is one
interpreter per case.

WHAT IT PINS (mesh slice DB2, from the cross-host matrix's finding): a peer's id
used to reach the LOCAL resolver first, so a conversation the sidebar lists
perfectly well answered "no session … to resume" from a shell while the TUI's own
``/resume`` opened it. Two sentences, one per case:

* an id ANOTHER DEVICE holds ⇒ the run names the DEVICE and the two ways in, and
  the local typo refusal is NOT printed. On a real terminal the same command opens
  the peer's session as a viewer (driven separately under a pty:
  ``$LOCAL_OPERATOR_SCRATCHPAD/shell_resume_transcript.py``, which reads the peer's
  own transcript row off the screen);
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
        # A REAL DEVICE HAS A CONFIGURED PROVIDER, and the CLI refuses to boot
        # without one BEFORE any session factory runs — so without this the cell
        # would measure the "not configured" banner instead of the messages it is
        # about (measured: the peer's case surfaced that banner once the pre-check
        # stopped refusing early). ``hosting: test`` is the harness's own provisioned
        # provider, so nothing here reaches a network.
        (root / "config.yml").write_text(
            "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n", encoding="utf-8"
        )

        # THE UNKNOWN ID: unchanged, and asserted FIRST so a regression in the new
        # arm cannot mask it (this cell runs the refusal the change was not allowed
        # to touch).
        code, out, err = await asyncio.to_thread(_run_cli, root, home, "--resume", "feedfacecafe")
        assert code == 1, (code, out, err)
        assert "no session 'feedfacecafe' to resume" in err, err
        assert "lives on" not in out, out

        # THE PEER'S ID: the run says WHERE IT LIVES rather than calling it a typo.
        #
        # THIS RUN HAS NO TERMINAL, and the sentence is the one for that case --
        # deliberately, because whether a viewer can be hosted is decided in
        # ``session_factory`` (on ``has_ui``, which is ``isatty``/``--tui``) and not
        # by the CLI's pre-check, which cannot know it. Measured with the promise
        # made in the pre-check instead: "opening it remotely" was printed and the
        # run then died with ``ResumeNotFound``. The pty rig named in this module's
        # docstring covers the terminal half end to end.
        code, out, err = await asyncio.to_thread(
            _run_cli, root, home, "--resume", created.session_id
        )
        assert f"{created.session_id} lives on " in err, (code, out, err)
        assert "no full-screen front end" in err, (code, out, err)
        assert "--engage " + created.session_id in err, (code, out, err)
        # AND THE LOCAL REFUSAL IS NOT PRINTED: that sentence is the defect this
        # arm exists to remove, and it would otherwise arrive from the resolver one
        # line later.
        assert "to resume" not in err, (
            "a conversation another device holds was answered as a typo: " + err
        )
    finally:
        await asyncio.to_thread(created.stop)
