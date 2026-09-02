"""The install-on-demand chokepoint: a no-op stub with a settled contract."""

from __future__ import annotations

from pathlib import Path

from local_operator.wakes.install import (
    NOT_AVAILABLE_REASON,
    InstallOutcome,
    ensure_supervisor_installed,
)


def test_stub_reports_not_installed_and_touches_nothing(tmp_path: Path) -> None:
    outcome = ensure_supervisor_installed(tmp_path)
    assert outcome == InstallOutcome(installed=False, reason=NOT_AVAILABLE_REASON)
    assert list(tmp_path.iterdir()) == []


def test_stub_is_idempotent(tmp_path: Path) -> None:
    first = ensure_supervisor_installed(tmp_path)
    second = ensure_supervisor_installed(tmp_path)
    assert first == second
