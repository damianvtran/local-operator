"""The evidence store on a platform with no POSIX at all (audit D5 / B19).

The first defect was not that the store lacked a platform guard — it has a
complete one, ``EvidenceWriter._supported``, whose message already names the
reason. The defect was that ``import fcntl`` sat at module scope, so nothing on
Windows ever reached the guard: the import of this module failed first, and
because ``evaluation.runner.episode`` imports it at module scope, the whole
evaluation runner was unimportable there.

The second defect was the SAME class one line away, and this file's first fix
did not catch it: ``os.register_at_fork`` is POSIX-only and was called bare, so
the module stayed unimportable on Windows — this time on an attribute rather
than a module — and the real Windows runner is what found it
(``xplat-probe-windows``: ``AttributeError: module 'os' has no attribute
'register_at_fork'``). The lesson is in the harness below: simulating only the
one module the last defect used is how the next defect in the class ships, so
the child interpreter is now stripped of EVERY POSIX-only module and attribute
the tree can reach, not just the one that bit.

So the assertions below are about ORDER: the module must import, and the
refusal a caller then meets must be the guard's own sentence. A blocked module
rather than a deleted attribute for the imports, because the import machinery is
what has to fail — a ``fcntl = None`` stand-in would not reproduce the
``ModuleNotFoundError`` the audit recorded.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]

#: Modules that exist only where POSIX does. Kept in step with the battery's own
#: list (``scripts/xplat_probe.py``'s ``POSIX_ONLY_MODULES``) so the simulation
#: and the instrument that grades the platform agree on what "POSIX-only means".
_POSIX_ONLY_MODULES = ("fcntl", "termios", "pwd", "grp", "pty", "resource", "tty", "crypt")

#: ``os`` attributes that are absent on Windows. Anything this tree reaches at
#: module scope OR inside a function that a POSIX-only caller guards will show up
#: here as an AttributeError the moment that guard slips — which is the canary.
#:
#: NOT ``os.kill``: it exists on Windows, and it is the destructive difference
#: (``os.kill(pid, 0)`` terminates the process it probes) rather than an absence
#: that makes it a defect. It is caught by the shared ``procstate.pid_liveness``
#: tests, not by deleting the attribute here.
_POSIX_ONLY_OS_ATTRIBUTES = (
    "register_at_fork",
    "fork",
    "forkpty",
    "killpg",
    "getuid",
    "geteuid",
    "getgid",
    "getegid",
    "getpgid",
    "getsid",
    "setsid",
    "setpgid",
    "wait3",
    "wait4",
    "chroot",
    "chown",
    "lchown",
    "nice",
    "getloadavg",
    "sysconf",
    "O_CLOEXEC",
    "O_DIRECTORY",
    "O_NOFOLLOW",
    "O_NONBLOCK",
)

#: ``signal`` attributes that are absent on Windows. Deliberately excludes
#: SIGINT/SIGTERM/SIGABRT/SIGSEGV/SIGILL/SIGFPE, which the ``signal`` module
#: defines there — deleting a portable one would be a fake failure, and a
#: simulation that fails for the wrong reason is worse than none.
_POSIX_ONLY_SIGNAL_ATTRIBUTES = (
    "SIGKILL",
    "SIGUSR1",
    "SIGUSR2",
    "SIGWINCH",
    "SIGSTOP",
    "SIGCHLD",
    "SIGPIPE",
    "SIGALRM",
    "SIGVTALRM",
    "SIGPROF",
    "SIGPOLL",
    "SIGBUS",
    "SIGSYS",
    "SIGHUP",
    "SIGQUIT",
    "SIGCONT",
    "SIGTSTP",
    "SIGTTIN",
    "SIGTTOU",
    "SIGURG",
    "alarm",
    "setitimer",
    "getitimer",
    "sigwait",
    "sigwaitinfo",
    "sigtimedwait",
    "sigpending",
    "pthread_kill",
    "pthread_sigmask",
)


def _posix_absent_preamble() -> str:
    """Source for a child interpreter whose platform looks like Windows.

    Generated rather than a literal so the two tuples above are the single
    source of truth for what "POSIX-only" means in this file.
    """
    return (
        "import os, signal, sys\n"
        "\n"
        "class _Blocked:\n"
        "    def __init__(self, names):\n"
        "        self._names = set(names)\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.partition('.')[0] in self._names:\n"
        '            raise ModuleNotFoundError(f"No module named {name!r}")\n'
        "        return None\n"
        "\n"
        f"sys.meta_path.insert(0, _Blocked({_POSIX_ONLY_MODULES!r}))\n"
        "for _name in ("
        f"{_POSIX_ONLY_MODULES!r}):\n"
        "    sys.modules.pop(_name, None)\n"
        f"for _name in {_POSIX_ONLY_OS_ATTRIBUTES!r}:\n"
        "    if hasattr(os, _name):\n"
        "        delattr(os, _name)\n"
        f"for _name in {_POSIX_ONLY_SIGNAL_ATTRIBUTES!r}:\n"
        "    if hasattr(signal, _name):\n"
        "        delattr(signal, _name)\n"
    )


def _run(tmp_path: Path, statement: str) -> subprocess.CompletedProcess[str]:
    """Run ``statement`` in a fresh interpreter with no POSIX-only surface left.

    The guards below are asserted rather than assumed: a preamble that silently
    stopped deleting ``os.register_at_fork`` (a CPython rename, a ``delattr`` on
    something already absent) would make every test in this file vacuous, and a
    vacuous pass is the failure mode that let the second defect through.
    """
    environment = {key: value for key, value in os.environ.items() if not key.startswith("CMUX_")}
    environment.update(
        HOME=str(tmp_path / "home"),
        LOCAL_OPERATOR_CONFIG_DIR=str(tmp_path / "config"),
        PYTHONPATH=str(REPO_ROOT),
    )
    (tmp_path / "home").mkdir(exist_ok=True)
    (tmp_path / "config").mkdir(exist_ok=True)
    guard = (
        "assert not hasattr(os, 'register_at_fork'), 'harness left os.register_at_fork in place'\n"
        "assert not hasattr(signal, 'SIGKILL'), 'harness left signal.SIGKILL in place'\n"
        "try:\n"
        f"    __import__({_POSIX_ONLY_MODULES[0]!r})\n"
        "except ModuleNotFoundError:\n"
        "    pass\n"
        "else:\n"
        f"    raise AssertionError('harness did not block {_POSIX_ONLY_MODULES[0]}')\n"
    )
    return subprocess.run(
        [sys.executable, "-c", _posix_absent_preamble() + guard + statement],
        capture_output=True,
        text=True,
        env=environment,
        timeout=180,
    )


def test_the_harness_actually_removes_the_posix_surface(tmp_path: Path) -> None:
    """The simulation is asserted, not trusted.

    Without this, a preamble that stopped deleting an attribute would turn every
    test below into an assertion about a normal POSIX interpreter — green, and
    blind to exactly the class of defect it exists to catch.
    """
    result = _run(
        tmp_path,
        "import os, signal\n"
        "assert not hasattr(os, 'register_at_fork')\n"
        "assert not hasattr(os, 'getuid')\n"
        "assert not hasattr(os, 'O_NONBLOCK')\n"
        "assert not hasattr(signal, 'SIGKILL')\n"
        "print('stripped')\n",
    )
    assert result.returncode == 0, result.stderr
    assert "stripped" in result.stdout


def test_the_evidence_store_imports_where_posix_does_not_exist(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        "from local_operator.evaluation.evidence.store import EvidenceWriter\n"
        "print('imported', EvidenceWriter.__name__)\n",
    )
    assert result.returncode == 0, result.stderr
    assert "imported EvidenceWriter" in result.stdout


def test_the_refusal_is_the_guards_own_sentence_not_an_import_error(tmp_path: Path) -> None:
    """The public entry point, reached on such a platform, refuses BY NAME.

    ``_supported`` is called before any syscall, so the caller gets the
    documented ``EvidenceUnsupported`` rather than a traceback from inside the
    locking helper — which is what makes the dead-code guard live again.
    """
    bundle = tmp_path / "bundle"
    result = _run(
        tmp_path,
        "from pathlib import Path\n"
        "from tests.unit.evaluation.evidence.test_models import manifest\n"
        "from tests.unit.evaluation.evidence.test_store import redactions\n"
        "from local_operator.evaluation.evidence.store import EvidenceUnsupported, EvidenceWriter\n"
        "try:\n"
        f"    EvidenceWriter.create(Path({str(bundle)!r}), manifest(), redactions('secret'))\n"
        "except EvidenceUnsupported as exc:\n"
        "    print('REFUSED:', exc)\n"
        "else:\n"
        "    print('NOT REFUSED')\n",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("REFUSED:"), result.stdout + result.stderr
    assert "POSIX flock and directory descriptors" in result.stdout
