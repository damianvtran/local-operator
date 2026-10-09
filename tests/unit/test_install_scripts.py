"""The one-line installers: what a checkout can actually check about them.

These are shells, not Python — the behaviour (phases, ETAs, elapsed, the
installed CLI answering) is exercised by running them, which CI's xplat probe
does on a real host. What a unit test can pin is the set of invariants that are
pure text and, when broken, break silently on somebody else's machine.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SH = ROOT / "scripts" / "install.sh"
PS1 = ROOT / "scripts" / "install.ps1"


def test_install_ps1_is_pure_ascii_and_bomless() -> None:
    """Review round 1, R-6: PowerShell 5.1 decodes a BOM-less file as ANSI.

    The header advertises 5.1+, so a `✓` or a `·` in the script printed as
    mojibake on `powershell -File scripts/install.ps1` — the documented
    `irm … | iex` path was fine (the HTTP charset decides that decode), which is
    why nobody saw it. A BOM would also fix 5.1, but it survives into the string
    `iex` receives on the documented path, so ASCII is the form that cannot
    break either route.
    """
    raw = PS1.read_bytes()
    assert raw[:3] != b"\xef\xbb\xbf", "a BOM would be passed on to `iex`"
    offenders = sorted({byte for byte in raw if byte > 0x7F})
    assert offenders == [], f"non-ASCII bytes in install.ps1: {offenders}"


def test_neither_installer_pins_a_version() -> None:
    """PRs never carry a version/never hard-code one (AGENTS.md, Releases).

    Both scripts install the package by NAME and print whatever version the
    installed CLI reports; a literal version here is how a stale installer
    ships an old build to a new user.
    """
    for path in (SH, PS1):
        text = path.read_text(encoding="ascii" if path is PS1 else "utf-8")
        assert not re.search(r"local-operator==\d", text), path
        assert "LOCAL_OPERATOR_PACKAGE" in text, path


def test_install_sh_states_its_python_requirement_and_reuses_one() -> None:
    """Q1: the uv-absent leg's cost was uv downloading CPython.

    The script must pass a Python it FOUND to `uv tool install` when the machine
    already has a 3.12+, and fall back to uv's managed interpreter when it does
    not — so the ETA line has to say which arm the user is on.
    """
    text = SH.read_text(encoding="utf-8")
    assert "find_python()" in text
    assert '--python "$PYTHON_BIN"' in text
    assert '--python "$PYTHON_VERSION"' in text
    # Both arms announce themselves, so the ETA matches the work.
    assert "Installing Local Operator (using " in text
    assert "uv brings Python" in text
