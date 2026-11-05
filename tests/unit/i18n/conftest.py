"""Fixtures loading the `scripts/i18n/` tools as importable modules.

They are scripts (stdlib-only entry points), so tests load them by path with
`sys.modules` registration (dataclasses need the module present under its own
name). Generator/watch instances are session-scoped. The checker is function-
scoped so its explicit synthetic-root structural adapter cannot leak to another
case or depend on a shorter-lived monkeypatch from a session fixture.
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


@pytest.fixture(autouse=True)
def _local_cli_invocation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run these unit fixtures as a local CLI, never as a GHA merged checkout.

    A unit shard exports its own pull-request provenance for the whole job, but
    the repositories these tests drive are synthetic (or the real tree checked
    out for a unit run, not the runner's merge commit). The strict merged-
    checkout proof is right to reject that mismatch, so inheriting the ambient
    provenance would only ever assert the fixture is not what CI checked out.
    Remove it here and let each test state the contract it exercises; the real
    provenance proof lives in test_audit_history's explicit environments and in
    the dedicated i18n job, which invokes the unchanged script directly.
    """
    for name in (
        "GITHUB_ACTIONS",
        "GITHUB_EVENT_NAME",
        "GITHUB_EVENT_PATH",
        "GITHUB_REF",
        "GITHUB_SHA",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def i18n_check(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    checker = _load("i18n_check_under_test", REPO / "scripts" / "i18n" / "check.py")
    normal_audit_gate = checker.check_audit_history

    def fixture_audit_gate(base=None, *, structural_only=False):
        # Existing ratchet cases deliberately construct repo-less mini-roots.
        # Model ONLY those invocations as structural fixtures, without clearing
        # CI/event environment globally or weakening the actual CLI's base gate.
        # Real checkout/Git-root calls retain the unmodified production gate;
        # test_audit_history also imports production check directly for its CI
        # negative controls, independent of this fixture adapter.
        if (
            checker.REPO.resolve() == REPO.resolve()
            or checker._git("rev-parse", "--is-inside-work-tree").returncode == 0
        ):
            return normal_audit_gate(base, structural_only=structural_only)
        from scripts.i18n.ledger import load_entries

        try:
            load_entries(checker.REPO)
        except (ValueError, OSError) as exc:
            return [f"audit history: {exc}"]
        return []

    monkeypatch.setattr(checker, "check_audit_history", fixture_audit_gate)
    return checker


@pytest.fixture(scope="session")
def i18n_generate() -> ModuleType:
    return _load("i18n_generate_under_test", REPO / "scripts" / "i18n" / "generate.py")


@pytest.fixture(scope="session")
def i18n_deferred_watch() -> ModuleType:
    return _load("i18n_deferred_watch_under_test", REPO / "scripts" / "i18n" / "deferred_watch.py")
