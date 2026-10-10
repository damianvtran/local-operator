"""Shared helpers for the turn-supplement contract tests (lane C0)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "supplements"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


def load(relative: str) -> Any:
    return json.loads((FIXTURES / relative).read_text(encoding="utf-8"))


def raw(relative: str) -> str:
    return (FIXTURES / relative).read_text(encoding="utf-8")
