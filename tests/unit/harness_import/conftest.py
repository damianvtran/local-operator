"""Isolated HOME/config fixtures for harness-import tests.

Every test copies the checked-in SYNTHETIC fixture under its own temporary
HOME. No test points an adapter at the operator's real ``~/.claude``,
``~/.codex`` or ``~/.omp``: these tests must stay hermetic, and a committed
fixture is the only place fake conversation text belongs.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    config = tmp_path / "config"
    home.mkdir()
    config.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    for name in (
        "CLAUDE_CONFIG_DIR",
        "CODEX_HOME",
        "PI_CODING_AGENT_DIR",
        "PI_CONFIG_DIR",
        "OMP_PROFILE",
        "PI_PROFILE",
    ):
        monkeypatch.delenv(name, raising=False)
    return home, config


@pytest.fixture
def claude_store(isolated):
    home, config = isolated
    shutil.copytree(FIXTURES / "claude", home / ".claude")
    return home, config


@pytest.fixture
def omp_store(isolated):
    home, config = isolated
    target = home / ".omp" / "agent"
    target.mkdir(parents=True)
    shutil.copytree(FIXTURES / "omp" / "sessions", target / "sessions")
    return home, config


@pytest.fixture
def codex_store(isolated):
    home, config = isolated
    sessions = home / ".codex" / "sessions" / "2026" / "01" / "01"
    sessions.mkdir(parents=True)
    shutil.copy2(FIXTURES / "codex" / "rollout-synthetic.jsonl", sessions)
    shutil.copy2(FIXTURES / "codex" / "rollout-rollback.jsonl", sessions)
    return home, config
