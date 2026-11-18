"""``python -m local_operator`` is a real launch route, not an accident.

``update_bootstrap.controller`` runs both the controller preparation and the
post-flip rolling-update tail as ``<python> -B -P -m local_operator update ...``.
Before ``local_operator/__main__.py`` existed that launch died with "No module
named local_operator.__main__; 'local_operator' is a package and cannot be
directly executed", so every rolling update reported "central services remain
pending" and never refreshed services, rolled out or pruned old installs. These
cells launch the module for real (unit tests that stub ``_run`` cannot see it).
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import local_operator
from local_operator import update_bootstrap as bootstrap
from local_operator.install_defaults import STABLE_ROOT_RELATIVE

# The tree the test process imported, so the child runs the same source. Every
# launch runs with cwd=tmp_path (a NEUTRAL directory holding no checkout), and
# production's ``-P`` keeps cwd off sys.path anyway, so the ONLY route to the
# package is this entry: the same one the parent imported through (editable or
# installed site-packages, or an explicit PYTHONPATH in a bounded rig).
_REPO_ROOT = Path(local_operator.__file__).resolve().parents[1]
# A cold CLI import plus parser build; generous so a loaded CI runner cannot
# flake it, while a hang still fails instead of wedging the shard.
_LAUNCH_TIMEOUT_S = 120


def _isolated_env(tmp_path: Path) -> dict[str, str]:
    """A child env that cannot touch the operator's config, home or cmux panes."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("CMUX_")}
    home = tmp_path / "home"
    config = tmp_path / "config"
    home.mkdir()
    config.mkdir()
    env.update(
        HOME=str(home),
        LOCAL_OPERATOR_CONFIG_DIR=str(config),
        PYTHONPATH=str(_REPO_ROOT),
        PYTHONDONTWRITEBYTECODE="1",
    )
    return env


def _launch(argv: list[str], tmp_path: Path, env: dict[str, str]):
    return subprocess.run(
        argv,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=_LAUNCH_TIMEOUT_S,
        check=False,
    )


def test_main_module_is_importable_as_a_spec():
    assert importlib.util.find_spec("local_operator.__main__") is not None


def test_python_dash_m_local_operator_reports_the_version(tmp_path: Path):
    env = _isolated_env(tmp_path)
    result = _launch([sys.executable, "-m", "local_operator", "--version"], tmp_path, env)
    assert result.returncode == 0, result.stderr
    # The expected value comes from the SAME import route in a child, so an
    # editable install whose metadata points at another checkout cannot skew it.
    expected = _launch(
        [
            sys.executable,
            "-c",
            "from local_operator.update import installed_version; print(installed_version())",
        ],
        tmp_path,
        env,
    )
    assert expected.returncode == 0, expected.stderr
    assert result.stdout.strip() == f"v{expected.stdout.strip()}"


def test_controller_module_command_launches_the_update_parser(tmp_path: Path):
    # The production builder both controller launches use, run for real with
    # ``--help`` so it proves the ``-B -P -m local_operator`` route and the
    # ``update`` parser without any update side effect.
    argv = bootstrap._module_command(Path(sys.executable), "update", "--help")
    assert argv[1:5] == ["-B", "-P", "-m", "local_operator"]
    result = _launch(argv, tmp_path, _isolated_env(tmp_path))
    assert "No module named local_operator.__main__" not in result.stderr
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_controller_tail_argv_reaches_the_tail_guard(tmp_path: Path):
    # The EXACT argv ``controller`` spawns after the core flip (same builder,
    # same flags), launched with no ``--help``. In an isolated HOME there is
    # no generation store, so the tail must pass module import and the
    # ``update`` parser, dispatch to ``controller_tail`` and stop at its first
    # guard: that guard's message is the proof the regressed hop now works,
    # and it fires before any mutation, so the run has no side effect.
    env = _isolated_env(tmp_path)
    promoted = tmp_path / "home" / "no-such-generation"
    argv = bootstrap._tail_command(Path(sys.executable), promoted, "0.0.0", no_services=True)
    assert argv[1:9] == [
        "-B",
        "-P",
        "-m",
        "local_operator",
        "update",
        "--controller-tail",
        str(promoted),
        "--controller-target",
    ]
    assert argv[-1] == "--no-services"
    result = _launch(argv, tmp_path, env)
    assert "No module named local_operator.__main__" not in result.stderr
    assert "usage:" not in result.stderr, result.stderr  # parser accepted the argv
    assert result.returncode == 1, (result.stdout, result.stderr)
    assert (
        "rolling update: service tail is not executing a canonical core generation" in result.stderr
    )
    # No generation store appeared in the isolated home: the guard fired before
    # the tail touched the install layout. (Not "home is empty": CLI startup may
    # legitimately create caches there, which says nothing about the tail.)
    assert not (tmp_path / "home" / STABLE_ROOT_RELATIVE).exists()
    assert not promoted.exists()
