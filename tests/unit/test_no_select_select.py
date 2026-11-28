"""Keep ``select.select`` out of code that runs inside a long-lived process.

Why: ``select.select`` cannot represent a descriptor at or above FD_SETSIZE
(1024) and raises ``ValueError: filedescriptor out of range in select()``. An
xdist worker accumulates descriptors across thousands of tests, so a readiness
check that is fine in isolation fails once the worker is deep enough -- PR
#4472's CI run 45338434283 (job 138256298160, ``test (3.13, 1)``, gw0) lost
seven ``test_linux_prepared_writer.py`` tests that way. The product is a
long-lived process too. ``tests/fd_readiness.py`` is the ceiling-free
replacement for tests; product code uses ``selectors``.

Scope: ``local_operator/`` and ``tests/unit/`` (including scripts embedded as
strings, which is why this is a text scan rather than an AST walk). ``scripts/``
and ``tests/e2e/`` are left alone on purpose: they run as their own short-lived
processes with low descriptor numbers.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

_CALL = re.compile(r"\bselect\.select\(|\bfrom\s+select\s+import\s+[^\n]*\bselect\b(?!\w)")
#: ``import select as sel`` -- the alias's ``.select(`` is the same call under
#: another name, so each alias found in a file is scanned for as well.
_ALIAS = re.compile(r"\bimport\s+select\s+as\s+(\w+)")

#: Repo-relative path -> (exact number of raw calls, why they are right there).
#: The count is pinned so an allowlisted file cannot quietly grow a second,
#: unguarded call next to its reviewed one. Non-test entries must keep their
#: ``_SELECT_FD_LIMIT`` guard (asserted below), so an allowlisted non-test call
#: only ever sees a descriptor select() can represent.
ALLOWED: dict[str, tuple[int, str]] = {
    "tests/fd_readiness.py": (
        5,
        "the replacement itself: select below _SELECT_FD_LIMIT keeps the old answer "
        "exactly (macOS reports FIFO EOF only via select), a selector above it",
    ),
    "tests/unit/test_fd_readiness.py": (
        2,
        "negative control: proves the raw call still raises above fd 1024, "
        "so the high-fd helper tests are not passing vacuously",
    ),
    "local_operator/providers/auth_cli.py": (
        1,
        "reads a tty; select is used only below _SELECT_FD_LIMIT, poll above it, "
        "because macOS poll() mishandles character devices (see _readable_within)",
    ),
    "local_operator/tui/host_theme.py": (
        1,
        "probes /dev/tty; same split as auth_cli: select only below "
        "_SELECT_FD_LIMIT, poll above it (see _readable_within)",
    ),
}


def _scanned_files() -> list[Path]:
    files = [
        *(REPO / "local_operator").rglob("*.py"),
        *(REPO / "tests" / "unit").rglob("*.py"),
        REPO / "tests" / "fd_readiness.py",
    ]
    return sorted(path for path in files if path.resolve() != Path(__file__).resolve())


def _call_offsets(text: str) -> list[int]:
    """Start offsets of every raw ``select.select`` spelling in ``text``."""
    offsets = [match.start() for match in _CALL.finditer(text)]
    for alias in {match.group(1) for match in _ALIAS.finditer(text)}:
        aliased = re.compile(rf"\b{re.escape(alias)}\.select\(")
        offsets.extend(match.start() for match in aliased.finditer(text))
    return sorted(offsets)


def find_offenders(paths: list[Path], root: Path) -> list[str]:
    """``path:line`` for each ``select.select`` use outside the allowlist."""
    offenders: list[str] = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if relative in ALLOWED:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for offset in _call_offsets(text):
            line = text.count("\n", 0, offset) + 1
            offenders.append(f"{relative}:{line}")
    return offenders


def test_no_select_select_outside_the_allowlist() -> None:
    offenders = find_offenders(_scanned_files(), REPO)
    assert not offenders, (
        "select.select() raises ValueError for fds >= 1024, which a long-lived "
        "process (an xdist worker, the product) reaches. Use tests/fd_readiness.py "
        "in tests or selectors in product code. Offenders: " + ", ".join(offenders)
    )


def test_allowlisted_files_still_guard_the_fd_limit() -> None:
    for relative, (expected, reason) in ALLOWED.items():
        text = (REPO / relative).read_text(encoding="utf-8")
        found = len(_call_offsets(text))
        assert found, f"{relative} no longer calls select.select; drop it ({reason})"
        assert found == expected, (
            f"{relative} has {found} raw select.select call(s), allowlisted for {expected}: "
            "review the new one for the FD_SETSIZE guard, then update the count"
        )
        if not relative.startswith("tests/unit/"):
            assert "< _SELECT_FD_LIMIT" in text, f"{relative} lost its FD_SETSIZE guard"


def test_the_scan_catches_each_spelling(tmp_path: Path) -> None:
    """Negative control: the pattern must fire, or the guard above proves nothing."""
    samples = {
        "call.py": "import select\nselect.select([fd], [], [], 0)\n",
        "embedded.py": 'SCRIPT = """\nready = select.select(rs, [], [], 1)\n"""\n',
        "imported.py": "from select import select\n",
        "aliased.py": "import select as sel\nready = sel.select([fd], [], [], 0)\n",
        "clean.py": "import selectors\nselectors.DefaultSelector()\nselect.PIPE_BUF\n",
    }
    for name, body in samples.items():
        (tmp_path / name).write_text(body)
    offenders = find_offenders(sorted(tmp_path.iterdir()), tmp_path)
    assert offenders == ["aliased.py:2", "call.py:2", "embedded.py:2", "imported.py:1"]
