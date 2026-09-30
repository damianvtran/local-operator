"""Shared fixtures: every hub_sync test runs against a private config dir, never the operator's."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in list(os.environ):
        if name.startswith(("CMUX_", "LOP_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / ".local-operator"))
    return tmp_path


class FakeClient:
    """A stand-in Radient client: ``get_team`` and agent zips are served from dicts."""

    def __init__(self, teams: dict[str, Any] | None = None) -> None:
        self.teams = teams or {}
        self.calls: list[str] = []

    def get_team(self, team_id: str, **_kw: Any) -> dict[str, Any]:
        self.calls.append(team_id)
        if team_id not in self.teams:
            error = RuntimeError("404 Not Found")
            error.status_code = 404  # type: ignore[attr-defined]
            raise error
        return dict(self.teams[team_id])
