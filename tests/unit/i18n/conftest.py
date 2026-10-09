"""Fixtures loading the `scripts/i18n/` tools as importable modules.

They are scripts (stdlib-only entry points), so tests load them by path with
`sys.modules` registration (dataclasses need the module present under its own
name). One instance per session; tests patch the module's globals with
`monkeypatch.setattr`, which restores them automatically.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[3]


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def i18n_check() -> ModuleType:
    return _load("i18n_check_under_test", REPO / "scripts" / "i18n" / "check.py")


@pytest.fixture(scope="session")
def i18n_generate() -> ModuleType:
    return _load("i18n_generate_under_test", REPO / "scripts" / "i18n" / "generate.py")
