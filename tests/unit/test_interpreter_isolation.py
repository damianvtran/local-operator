"""A spawned child must import the INSTALL, never a checkout that is merely the cwd.

WHY THESE ASSERT ON THE IMPORTED TREE, NOT ON THE ARGV
------------------------------------------------------
``-P`` in an argv list is the mechanism; the behaviour is which
``local_operator/__init__.py`` the child actually binds. A test that only
checks the flag is present passes just as happily if CPython changes what the
flag does, or if a later edit reorders it after ``-m`` where the interpreter
stops recognising it. So each test here spawns a REAL child from a directory
containing a decoy package and asserts on the path the child reports importing.

The decoy is a fake ``local_operator`` package rather than a real checkout:
building one would cost a clone per test, and the property under test is only
"does the cwd shadow site-packages", which any importable directory of that
name demonstrates exactly as well.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from local_operator.interpreter import SAFE_PATH_FLAG, python_argv

#: Reports which tree the name resolved to, and nothing else, so a failure
#: message names the offending path rather than merely saying "False".
_PROBE = "import local_operator; print(local_operator.__file__)"


def _decoy_tree(root: Path) -> Path:
    """A directory holding a package that would shadow the real one.

    Marked with a sentinel attribute so a child importing it is unmistakable —
    a wrong path could otherwise be misread as a venv layout difference.
    """
    package = root / "local_operator"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("DECOY = True\n", encoding="utf-8")
    return root


def _isolated_env() -> dict[str, str]:
    """Environment for a child that must not touch the developer's real state.

    Every ``CMUX_*`` variable is dropped: an inherited workspace id has
    previously let a headless test rename a real cmux workspace.
    """
    env = {key: value for key, value in os.environ.items() if not key.startswith("CMUX_")}
    # Must not leak in and mask the very difference these tests measure.
    env.pop("PYTHONSAFEPATH", None)
    return env


def _run(argv: list[str], cwd: Path) -> str:
    result = subprocess.run(
        argv, cwd=str(cwd), env=_isolated_env(), capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, f"child failed: {result.stderr}"
    return result.stdout.strip()


def test_bare_dash_m_imports_the_cwd_and_is_the_defect(tmp_path: Path) -> None:
    """The defect itself, pinned so the fix below is measured against something.

    This is what every spawn site did before: no ``cwd=``, so the child inherits
    the parent's directory, and ``-m``/``-c`` put that directory on
    ``sys.path[0]`` AHEAD of site-packages. On a real install this made a
    runtime import a 0.51.0 worktree while the install was 0.51.5.
    """
    root = _decoy_tree(tmp_path)
    imported = _run([sys.executable, "-c", _PROBE], root)
    assert imported == str(root / "local_operator" / "__init__.py"), (
        "expected the unprotected child to import the cwd decoy; if this fails the "
        "premise of this module changed and the fix below proves nothing"
    )


def test_python_argv_child_ignores_a_package_in_the_cwd(tmp_path: Path) -> None:
    """The fix: same interpreter, same cwd, resolves the installed package."""
    root = _decoy_tree(tmp_path)
    imported = _run(python_argv("-c", _PROBE), root)
    assert imported != str(root / "local_operator" / "__init__.py")
    # Positive assertion too: "not the decoy" would also be satisfied by an
    # unrelated failure mode, so name the tree it MUST have resolved to.
    import local_operator

    assert Path(imported) == Path(local_operator.__file__)


def test_isolation_flag_precedes_the_module_selector() -> None:
    """``-P`` after ``-m`` is consumed by the module, not by the interpreter.

    Ordering is silent when wrong — the child still starts, still runs, and
    still imports the checkout — so it is asserted rather than left to review.
    """
    argv = python_argv("-m", "local_operator.cli")
    assert argv[0] == sys.executable
    assert argv[1] == SAFE_PATH_FLAG
    assert argv.index(SAFE_PATH_FLAG) < argv.index("-m")


def test_isolation_does_not_leak_into_grandchildren(tmp_path: Path) -> None:
    """Why the FLAG and not ``PYTHONSAFEPATH=1`` in the child's env.

    The variable is inherited by the whole process subtree; the flag is not. A
    runtime spawned by these call sites goes on to run the agent's ``bash``
    tool, which copies ``os.environ`` into every command it executes — so the
    variable form would strip ``sys.path[0]`` from every ``python`` the USER
    runs in their own project, breaking ``import my_module`` beside a script.
    That is a total break of ordinary Python semantics in the user's workspace,
    caused by an isolation flag meant only to protect our own import.
    """
    root = _decoy_tree(tmp_path)
    grandchild = (
        "import subprocess, sys; "
        f"print(subprocess.run([sys.executable, '-c', {_PROBE!r}], "
        "capture_output=True, text=True).stdout.strip())"
    )
    # The flag protects the child but leaves the grandchild with normal rules,
    # so the grandchild still sees the cwd package.
    assert _run(python_argv("-c", grandchild), root) == str(root / "local_operator" / "__init__.py")

    # The variable form, by contrast, reaches the grandchild too.
    env = _isolated_env()
    env["PYTHONSAFEPATH"] = "1"
    leaked = subprocess.run(
        [sys.executable, "-c", grandchild],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert leaked.returncode == 0, leaked.stderr
    assert leaked.stdout.strip() != str(root / "local_operator" / "__init__.py")


def test_eval_worker_restores_cwd_imports_for_user_code(tmp_path: Path) -> None:
    """``-P`` must not cost the eval tool its workspace imports.

    The worker runs USER code, so a cell doing ``import my_module`` next to the
    session's cwd has to keep working. ``_enable_cwd_imports`` puts the cwd back
    AFTER the worker's own imports have resolved — this asserts both halves of
    that ordering at once: the harness module came from the install, and the
    user module still resolved from the cwd.
    """
    root = _decoy_tree(tmp_path)
    (root / "usermod.py").write_text("VALUE = 'from the workspace'\n", encoding="utf-8")
    probe = (
        "from local_operator.tools.eval_worker import _enable_cwd_imports; "
        "import local_operator; "
        "_enable_cwd_imports(); "
        "import usermod; "
        "print(local_operator.__file__); print(usermod.VALUE)"
    )
    out = _run(python_argv("-c", probe), root).splitlines()
    import local_operator

    assert Path(out[0]) == Path(local_operator.__file__), "harness must bind to the install"
    assert out[1] == "from the workspace", "user code must still import from the cwd"


def test_cwd_restore_appends_so_it_cannot_shadow_harness_imports(tmp_path: Path) -> None:
    """The restore appends rather than inserting at position 0.

    A stray module in a user's directory must not be able to take over a name
    the protocol itself imports; giving user code its workspace is not a licence
    to let it shadow the worker's own dependencies.
    """
    root = _decoy_tree(tmp_path)
    probe = (
        "from local_operator.tools.eval_worker import _enable_cwd_imports; "
        "_enable_cwd_imports(); "
        "import local_operator; print(local_operator.__file__)"
    )
    imported = _run(python_argv("-c", probe), root)
    assert imported != str(root / "local_operator" / "__init__.py")
