"""The ladder's MIDDLE RUNG at every self-spawn: a label with no image.

WHY THIS FILE EXISTS. The fallback rung of ``local_operator.procname`` is
"hardlink+symlink -> argv-only labelling -> exactly today's behaviour", and the
middle rung was missing at every spawn site: each one fell back to
``argv[0] = sys.executable`` when no branded image could be planted, so a machine
that cannot plant one (a framework interpreter, a system Python, a pip install
that is not a venv this project owns) showed a bare ``python3.x`` row for every
process the product starts.

The fix is a PAIRING rather than a decoration, and that is the second thing these
tests pin: on POSIX ``Popen(argv=[...])`` with ``executable=None`` EXECUTES
``argv[0]``, so a site that only relabelled argv[0] would ask the kernel to run a
file named ``Local Operator [session] id=...``. Every assertion below therefore
checks ``executable`` alongside the label — the two are only correct together.

Each test drives the REAL spawn function with the image probe forced to fail, and
reads what the spawn was actually called with.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from local_operator import procname
from local_operator.exec_mode import ExecArgs
from local_operator.paths import CONFIG_DIR_ENV


class _Child:
    """Enough of a ``Popen`` for the callers under test."""

    pid = 4321
    returncode = None

    def poll(self):
        return None

    def kill(self) -> None:
        return None


@pytest.fixture
def no_branded_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force rung 2: the image cannot be planted, the argv row still can."""
    monkeypatch.setattr(procname, "ensure_branded_interpreter", lambda: None)


def test_proc_spawn_detached_labels_and_supplies_the_image(
    no_branded_image, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator import proc

    recorded: dict[str, Any] = {}

    def fake_popen(argv, **kwargs):
        recorded["argv"] = list(argv)
        recorded["executable"] = kwargs.get("executable")
        return _Child()

    monkeypatch.setattr(proc.subprocess, "Popen", fake_popen)
    label = procname.branded_argv0(procname.LABEL_EXEC, job="job1234")
    started = proc.spawn_detached(
        [sys.executable, "-P", "-m", "local_operator.exec_worker"], label=label
    )

    assert started is True
    assert recorded["argv"][0] == label
    assert recorded["argv"][1:] == ["-P", "-m", "local_operator.exec_worker"]
    assert recorded["executable"] == sys.executable


def test_proc_spawn_detached_still_leaves_other_binaries_alone(
    no_branded_image, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The realpath guard: a terminal emulator must keep its own image.

    Without it, ``executable=`` would run PYTHON under the terminal's arguments
    — so this guard is not an optimisation, and relabelling must not widen it.
    """
    from local_operator import proc

    recorded: dict[str, Any] = {}

    def fake_popen(argv, **kwargs):
        recorded["argv"] = list(argv)
        recorded["executable"] = kwargs.get("executable")
        return _Child()

    monkeypatch.setattr(proc.subprocess, "Popen", fake_popen)
    proc.spawn_detached(["/usr/bin/open", "-a", "Ghostty"], label="Local Operator [fork]")

    assert recorded["argv"] == ["/usr/bin/open", "-a", "Ghostty"]
    assert recorded["executable"] is None


def test_spawn_runtime_labels_and_supplies_the_image(
    no_branded_image, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The viewer's detached runtime — the spawn ``launch.py`` owns."""
    from local_operator.interpreter import SAFE_PATH_FLAG
    from local_operator.session.runtime import launch as launch_module

    recorded: dict[str, Any] = {}

    def fake_popen(argv, **kwargs):
        recorded["argv"] = list(argv)
        recorded["executable"] = kwargs.get("executable")
        return _Child()

    monkeypatch.setattr(launch_module.subprocess, "Popen", fake_popen)
    process = launch_module._spawn_runtime("sess-fall01", str(tmp_path), defer_materialise=True)
    capture = getattr(process, "lop_capture_path", None)
    if capture is not None:
        capture.unlink(missing_ok=True)

    assert recorded["argv"][0] == "Local Operator [session] id=sess-fal"
    assert recorded["argv"][1] == SAFE_PATH_FLAG
    assert recorded["argv"][2:] == ["-m", "local_operator.session.runtime.process"]
    assert recorded["executable"] == sys.executable


def test_exec_background_labels_and_supplies_the_image(
    no_branded_image, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from local_operator import exec_mode

    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path / "config"))
    monkeypatch.setattr(exec_mode, "resolve_hosting_model_dry", lambda args: ("test", "m"))
    popen_mock = MagicMock()
    popen_mock.return_value.pid = 4321
    monkeypatch.setattr(exec_mode.subprocess, "Popen", popen_mock)
    monkeypatch.setattr(exec_mode, "_process_generation", lambda pid: None)

    assert (
        exec_mode.run_exec(
            "write a long report about penguins",
            ExecArgs(background=True, json_mode=True, yolo=True, hosting="openai"),
        )
        == 0
    )

    argv = popen_mock.call_args[0][0]
    assert argv[0].startswith("Local Operator [exec] job=")
    assert popen_mock.call_args[1]["executable"] == sys.executable


@pytest.mark.asyncio
async def test_eval_worker_labels_and_supplies_the_image(
    no_branded_image, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The eval worker — the one child whose argv this project most needs to read.

    An unexplained interpreter running USER code is the row an operator most
    needs to identify, and it is also the spawn whose cwd is deliberately the
    session's.
    """
    from local_operator.interpreter import SAFE_PATH_FLAG
    from local_operator.tools import eval as eval_tool

    class _Transport:
        closed = False

        def close(self) -> None:
            self.closed = True

    class _Process:
        pid = 4242
        returncode = None
        _transport = _Transport()

        async def wait(self) -> int:
            return 0

        def kill(self) -> None:
            return None

    recorded: dict[str, Any] = {}

    async def spawn(*args, **kwargs):
        recorded["argv"] = list(args)
        recorded["executable"] = kwargs.get("executable")
        return _Process()

    monkeypatch.setattr(eval_tool.asyncio, "create_subprocess_exec", spawn)
    kernel = await eval_tool._spawn(str(tmp_path))

    assert str(recorded["argv"][0]).startswith("Local Operator [eval] session=")
    assert recorded["argv"][1] == SAFE_PATH_FLAG
    assert recorded["argv"][2:] == ["-u", "-m", "local_operator.tools.eval_worker"]
    assert recorded["executable"] == sys.executable

    await eval_tool._close_kernel(kernel)


def test_the_installer_labels_and_supplies_the_image(no_branded_image) -> None:
    """``lop update``'s pip installer: a network install by an unnamed Python.

    Asserted here with the other spawn sites because it is the same invariant;
    ``test_update.py`` covers the argv the upgrade path builds.
    """
    from local_operator.update import InstallKind, installer_invocation

    argv, executable = installer_invocation(InstallKind.PIP)
    assert argv[0] == procname.branded_argv0(procname.LABEL_INSTALL)
    assert argv[1:] == ["-m", "pip", "install", "-U", "local-operator"]
    assert executable == sys.executable


def test_a_labelled_argv_really_starts(no_branded_image) -> None:
    """The rung is EXECUTED, not just described — the whole point of the pairing."""
    argv0, executable = procname.spawn_identity(procname.LABEL_DAEMONS_REFRESH)
    completed = subprocess.run(
        [argv0, "-c", "print('up')"],
        executable=executable,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "up"
