"""The naming ladder at every self-spawn: the label rides with the image.

WHY THIS FILE EXISTS, AND WHY IT NO LONGER CLAIMS WHAT IT FIRST DID. This
change first implemented the ladder's middle rung as "argv-only labelling": the
label as ``argv[0]`` on a machine with no plantable branded image, paired with
the bare interpreter as ``executable=`` so the kernel had something real to
execute. CI refused it, on ubuntu/py3.12 and nowhere else:

.. code-block:: text

    PermissionError: [Errno 13] Permission denied: ''

CPython on Linux derives ``sys.executable`` from ``argv[0]``, so a child born
with a LABEL in ``argv[0]`` has an EMPTY ``sys.executable`` — and
``secrets/client.py`` spawns the secret broker with ``executable=sys.executable``,
so the eval worker's own machinery died, along with any eval cell doing
``subprocess.run([sys.executable, …])``. macOS hides all of it: it resolves the
interpreter from the EXECUTED IMAGE, so the identical child reports a real
``sys.executable`` there.

So the ladder's middle rung is handled by rung 1 or not at all:
``procname.spawn_identity`` returns the label WITH the branded link, and
``(sys.executable, None)`` — no label applied — when no link can be planted.
These tests pin both halves of that: the parent-side pair at each spawn site,
AND the child-side consequence that forced the change, which is why the
child-side test is written to run for real on CI's ubuntu job rather than only
where the author could look at it.

The pairing is still load-bearing on the rung that does label: on POSIX
``Popen(argv=[...])`` with ``executable=None`` EXECUTES ``argv[0]``, so a label
without an image would ask the kernel to run a file named
``Local Operator [session] id=...``.

THE LAST EXCEPTION IS CLOSED HERE: ``secrets/client.py``'s broker spawn kept its
label with a plain image, justified by the broker reading ``sys.executable``
nowhere and spawning nothing (#1162's M-c). That is true of today's broker and is
not a contract to leave on a detached daemon whose failures are silent, so the
broker now resolves both axes through ``spawn_identity`` with every other site.

These tests drive the REAL spawn functions and read what the spawn was
actually called with. They take three shapes, because the ladder has three
distinct paths and only the first two are about the image probe:

- the **rung-2** tests force ``ensure_branded_interpreter`` to return ``None``
  (no image can be planted) and pin the unlabelled command line;
- the **rung-1** tests force it to return a real interpreter file, so both axes
  are labelled AND the child is genuinely executed through that pair;
- ``secrets/client.py``'s remaining tests stop asking the probe anything at all:
  they call the site's own ``_broker_identity``, or make ``spawn_identity``
  RAISE. The raising one is the ``decoration never fails a spawn`` contract,
  which is a different code path from ``no image to plant`` — an exception
  anywhere in the ladder has to land on the same command line rung 2 produces.
"""

from __future__ import annotations

import hashlib
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
    """Force rung 2: no image can be planted, so no label is applied either."""
    monkeypatch.setattr(procname, "ensure_branded_interpreter", lambda: None)


@pytest.fixture
def branded_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force rung 1 with a REAL image: any interpreter file will do."""
    monkeypatch.setattr(procname, "ensure_branded_interpreter", lambda: Path(sys.executable))


def test_spawn_identity_applies_no_label_without_an_image(no_branded_image) -> None:
    """The heart of the change, stated once for every caller below."""
    argv0, executable = procname.spawn_identity(procname.LABEL_SESSION_ANON, id="rung2")

    assert argv0 == sys.executable
    assert executable is None


def test_spawn_identity_labels_alongside_the_image(branded_image) -> None:
    argv0, executable = procname.spawn_identity(procname.LABEL_SESSION_ANON, id="rung1")

    assert argv0 == procname.branded_argv0(procname.LABEL_SESSION_ANON, id="rung1")
    assert executable == sys.executable


def test_the_fallback_rung_keeps_a_working_interpreter(no_branded_image) -> None:
    """The child-side property, on the rung CI's ubuntu job actually runs.

    THE REGRESSION TEST FOR THE EMPTY-``sys.executable`` BLOCKER, and it runs the
    interpreter rather than inspecting the spawn call: a parent-side assertion
    cannot see this failure, because the parent's argv and ``executable=`` look
    correct on both sides of it. A machine that cannot plant an image gets an
    unlabelled row — what it must never get is a child that cannot say which
    interpreter it is, because this product's own machinery (the secret broker)
    and user code in an eval cell both spawn through ``sys.executable``.
    """
    argv0, executable = procname.spawn_identity(procname.LABEL_EVAL, id="childtest")

    completed = subprocess.run(
        [argv0, "-c", "import sys; print(sys.executable or '')"],
        executable=executable,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    reported = completed.stdout.strip()
    assert reported, "the child reported an empty sys.executable"
    assert Path(reported).exists()


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason=(
        "rung 1 exists only where an image can be planted, and "
        "procname.branded_link_path() has no answer off darwin; the "
        "Linux-executed counterpart is the child test above"
    ),
)
def test_the_labelled_rung_runs_and_the_child_keeps_an_interpreter(branded_image) -> None:
    """Rung 1, executed end to end: the label, the image, and a working child."""
    argv0, executable = procname.spawn_identity(procname.LABEL_DAEMONS_REFRESH)
    assert argv0.startswith("Local Operator [")

    completed = subprocess.run(
        [argv0, "-c", "import sys; print(sys.executable or '')"],
        executable=executable,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    reported = completed.stdout.strip()
    assert reported and Path(reported).exists()


def test_proc_spawn_detached_leaves_the_interpreter_unlabelled(
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
    assert recorded["argv"] == [sys.executable, "-P", "-m", "local_operator.exec_worker"]
    assert recorded["executable"] is None


def test_proc_spawn_detached_labels_when_the_image_exists(
    branded_image, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of the same spawn: with an image, the label IS applied."""
    from local_operator import proc

    recorded: dict[str, Any] = {}

    def fake_popen(argv, **kwargs):
        recorded["argv"] = list(argv)
        recorded["executable"] = kwargs.get("executable")
        return _Child()

    monkeypatch.setattr(proc.subprocess, "Popen", fake_popen)
    label = procname.branded_argv0(procname.LABEL_EXEC, job="job1234")
    proc.spawn_detached([sys.executable, "-P", "-m", "local_operator.exec_worker"], label=label)

    assert recorded["argv"][0] == label
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


def test_spawn_runtime_leaves_the_interpreter_unlabelled(
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

    assert recorded["argv"][0] == sys.executable
    assert recorded["argv"][1] == SAFE_PATH_FLAG
    assert recorded["argv"][2:] == ["-m", "local_operator.session.runtime.process"]
    assert recorded["executable"] is None


def test_spawn_runtime_labels_when_the_image_exists(
    branded_image, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from local_operator.interpreter import SAFE_PATH_FLAG
    from local_operator.session.runtime import launch as launch_module

    recorded: dict[str, Any] = {}

    def fake_popen(argv, **kwargs):
        recorded["argv"] = list(argv)
        recorded["executable"] = kwargs.get("executable")
        return _Child()

    monkeypatch.setattr(launch_module.subprocess, "Popen", fake_popen)
    process = launch_module._spawn_runtime("sess-rung01", str(tmp_path), defer_materialise=True)
    capture = getattr(process, "lop_capture_path", None)
    if capture is not None:
        capture.unlink(missing_ok=True)

    assert recorded["argv"][0] == "Local Operator [session] id=sess-run"
    assert recorded["argv"][1] == SAFE_PATH_FLAG
    assert recorded["argv"][2:] == ["-m", "local_operator.session.runtime.process"]
    assert recorded["executable"] == sys.executable


def test_exec_background_leaves_the_interpreter_unlabelled(
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
    # ``tests/unit/test_exec_mode.py`` pins the other half of this contract —
    # ``executable=`` is set only when it names the branded image — which is the
    # invariant that was left contradictory before this round.
    assert argv[0] == sys.executable
    assert popen_mock.call_args[1]["executable"] is None


@pytest.mark.asyncio
async def test_eval_worker_leaves_the_interpreter_unlabelled(
    no_branded_image, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The eval worker — the one child whose argv this project most needs to read.

    An unexplained interpreter running USER code is the row an operator most
    needs to identify, and it is also the spawn whose cwd is deliberately the
    session's. It is also the process whose empty-``sys.executable`` broke the
    broker, so its rung-2 shape is pinned here and its child-side behaviour in
    ``test_the_fallback_rung_keeps_a_working_interpreter``.
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

    assert recorded["argv"][0] == sys.executable
    assert recorded["argv"][1] == SAFE_PATH_FLAG
    assert recorded["argv"][2:] == ["-u", "-m", "local_operator.tools.eval_worker"]
    assert recorded["executable"] is None

    await eval_tool._close_kernel(kernel)


@pytest.mark.asyncio
async def test_eval_worker_labels_when_the_image_exists(
    branded_image, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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


def test_the_installer_leaves_the_interpreter_unlabelled(no_branded_image) -> None:
    """``lop update``'s pip installer: a network install by an unnamed Python.

    Asserted here with the other spawn sites because it is the same invariant;
    ``test_update.py`` covers the argv the upgrade path builds.
    """
    from local_operator.update import InstallKind, installer_invocation

    argv, executable = installer_invocation(InstallKind.PIP)
    assert argv == [sys.executable, "-m", "pip", "install", "-U", "local-operator"]
    assert executable is None


def test_the_installer_labels_when_the_image_exists(branded_image) -> None:
    from local_operator.update import InstallKind, installer_invocation

    argv, executable = installer_invocation(InstallKind.PIP)
    assert argv[0] == procname.branded_argv0(procname.LABEL_INSTALL)
    assert argv[1:] == ["-m", "pip", "install", "-U", "local-operator"]
    assert executable == sys.executable


def test_the_broker_labels_alongside_the_image(
    branded_image, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The secret broker's image axis, which this spawn site did not set before.

    ``_spawn_broker`` passed the interpreter as BOTH the image and (pre-#958)
    argv, so a broker started from a plain interpreter parent — ``lop`` from a
    venv python, the common case — left Activity Monitor a row of ``python3.12``
    with the label only in argv. Resolving both axes through
    ``procname.spawn_identity`` is what makes the row the product's own name on
    the axis users read, and this pins the pair rather than the implementation.
    """
    from local_operator.secrets import client
    from local_operator.secrets.keys import secrets_dir

    recorded: dict[str, Any] = {}

    def fake_popen(argv, **kwargs):
        recorded["argv"] = list(argv)
        recorded["executable"] = kwargs.get("executable")
        return _Child()

    monkeypatch.setattr(client.subprocess, "Popen", fake_popen)
    base = tmp_path / "config"
    client._spawn_broker(base)

    digest = hashlib.sha256(str(secrets_dir(base)).encode("utf-8")).hexdigest()[:12]
    assert recorded["argv"] == [
        procname.branded_argv0(procname.LABEL_BROKER, digest=digest),
        "-m",
        "local_operator.secrets.brokerd",
    ]
    assert recorded["executable"] == sys.executable


def test_the_broker_leaves_the_interpreter_unlabelled(
    no_branded_image, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Rung 2 at THIS site, pinning the rule that used to have an exception here.

    The broker kept its label with a plain image — the last labelled-argv
    exception in the tree (#1162's M-c) — which costs the child its
    ``sys.executable`` on Linux. Where there is no image to ride with, the label
    goes: the exact pre-#958 command line, the interpreter with no
    ``executable=``.
    """
    from local_operator.secrets import client

    recorded: dict[str, Any] = {}

    def fake_popen(argv, **kwargs):
        recorded["argv"] = list(argv)
        recorded["executable"] = kwargs.get("executable")
        return _Child()

    monkeypatch.setattr(client.subprocess, "Popen", fake_popen)
    client._spawn_broker(tmp_path / "config")

    assert recorded["argv"] == [sys.executable, "-m", "local_operator.secrets.brokerd"]
    assert recorded["executable"] is None


def test_a_broker_branding_failure_falls_back_to_the_unlabelled_command_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Decoration never costs a broker its start — the module's absolute contract.

    Forced by making the branding itself fail rather than by removing the image,
    because the two paths are different code: the point of this test is that an
    exception anywhere in ``spawn_identity`` lands on the same command line
    rung 2 produces. A broker that cannot start is worse than an unnamed one.
    """

    def explode(*args: Any, **kwargs: Any):
        raise RuntimeError("branding blew up")

    monkeypatch.setattr(procname, "spawn_identity", explode)

    from local_operator.secrets import client

    recorded: dict[str, Any] = {}

    def fake_popen(argv, **kwargs):
        recorded["argv"] = list(argv)
        recorded["executable"] = kwargs.get("executable")
        return _Child()

    monkeypatch.setattr(client.subprocess, "Popen", fake_popen)
    client._spawn_broker(tmp_path / "config")

    assert recorded["argv"] == [sys.executable, "-m", "local_operator.secrets.brokerd"]
    assert recorded["executable"] is None


def test_a_broker_child_keeps_a_working_interpreter(no_branded_image, tmp_path: Path) -> None:
    """The child-side consequence, executed, at the site the blocker was found on.

    #1162's round-one blocker was this spawn site's OWN claim: the eval worker
    had an empty ``sys.executable`` and ``_spawn_broker`` was the code that died
    on it. The mechanism is Linux-only (``argv[0]`` → ``sys.executable``), and
    this test runs the broker's real identity pair rather than inspecting the
    call, so on CI's ubuntu job — where rung 2 is the only rung — it is the
    regression test; on macOS it runs rung 1 through the planted link and is
    true for the reason that hides the bug there.
    """
    from local_operator.secrets import client

    argv0, executable = client._broker_identity(tmp_path / "config")
    assert argv0 == sys.executable
    assert executable is None

    completed = subprocess.run(
        [argv0, "-c", "import sys; print(sys.executable or '')"],
        executable=executable,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    reported = completed.stdout.strip()
    assert reported, "the broker's command line left the child unable to name its interpreter"
    assert Path(reported).exists()
