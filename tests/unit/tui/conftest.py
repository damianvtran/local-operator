"""Hermetic environment for the TUI suite.

Snapshot frames were captured with colour enabled and shimmer off; a caller
that exports NO_COLOR (a common developer default) would otherwise fail all
three snapshots for reasons that have nothing to do with the code under
test. The pins live in fixtures — scoped and reverted — instead of module
import time, so collection order never leaks environment into other suites.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def hermetic_tui_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("LOCAL_OPERATOR_NO_SHIMMER", "1")
