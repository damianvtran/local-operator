"""The Linux comm axis: a process that names ITSELF.

WHY THIS IS A SEPARATE FILE. ``test_procname.py`` is macOS-only by design — the
branded-image mechanism it guards does not exist off darwin — while this half is
the opposite: ``prctl(PR_SET_NAME)`` is the only naming axis Linux has, since
there is no image to exec through (``branded_link_path`` returns ``None`` there).
Both files exist because CI runs the unit suite on both platforms and each half
is a no-op on the other.

WHY A SUBPROCESS. ``set_process_name`` renames the CALLING thread's ``comm``, and
a test that did that in-process would rename the pytest worker. Each assertion
therefore runs a child that names itself and reports what the kernel stored.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from local_operator import procname

pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="prctl is Linux-only; macOS names the process through its image",
)

REPORT = (
    "import pathlib, sys\n"
    "from local_operator import procname\n"
    "procname.{call}\n"
    "sys.stdout.write(pathlib.Path('/proc/self/comm').read_text().strip())\n"
)


def _comm_after(call: str, tmp_path: Path) -> str:
    script = tmp_path / "comm.py"
    script.write_text(REPORT.format(call=call))
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_brand_this_process_names_the_comm_axis(tmp_path: Path) -> None:
    assert _comm_after("brand_this_process()", tmp_path) == procname.BRAND


def test_cross_tree_spawn_withholds_the_label_without_an_image() -> None:
    """The cross-tree helper is rung 2 off macOS, and rung 2 is UNLABELLED.

    There is no image axis to plant on Linux, so the pair must be the bare
    target path with ``executable=None``. The reason it matters here more than
    on macOS is the same one that made the old argv-only rung a mistake: a
    labelled ``argv[0]`` leaves the child with an EMPTY ``sys.executable``,
    because CPython derives it from ``argv[0]`` (see the module ladder).
    """
    target = sys.executable
    argv0, executable = procname.spawn_identity_for_interpreter(
        procname.LABEL_DAEMONS_REFRESH, target
    )
    assert executable is None
    assert argv0 == target
    assert procname.BRAND not in argv0


def test_set_process_name_names_the_comm_axis(tmp_path: Path) -> None:
    assert _comm_after("set_process_name()", tmp_path) == procname.BRAND


def test_only_the_brand_fits_the_comm_window() -> None:
    """Linux ``comm`` is 15 bytes, so the label axis cannot live here.

    Recorded as a test rather than a comment because the temptation to pass the
    role label to ``prctl`` is real, and its symptom — ``Local Operator [`` in
    every row — is the same indistinguishable listing the branding exists to
    remove.
    """
    assert len(procname.BRAND.encode("utf-8")) <= 15
    assert len(procname.branded_argv0(procname.LABEL_MOBILE, port=4098).encode("utf-8")) > 15


def test_brand_this_process_never_raises(tmp_path: Path) -> None:
    """``comm`` is decoration: no failure of it may stop a session starting."""
    script = tmp_path / "no_raise.py"
    script.write_text(
        "from local_operator import procname\n"
        "procname.brand_this_process('anything')\n"
        "procname.brand_this_process()\n"
        "print('alive')\n"
    )
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "alive"
