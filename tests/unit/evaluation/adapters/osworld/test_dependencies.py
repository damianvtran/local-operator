"""Packaging failures are caught without executing task or parent packages."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from lop_osworld_v2_adapter import dependencies

_ROOT = Path(__file__).resolve().parents[5] / "benchmarks" / "osworld_v2_adapter"


def test_vendored_closure_is_exact_pinned_source() -> None:
    provenance = json.loads((_ROOT / "src/evaluation_examples/UPSTREAM.json").read_bytes())
    pin = json.loads((_ROOT / "config/release-v2026.08.08.json").read_bytes())
    assert provenance["commit"] == pin["osworld"]["commit"]
    files = provenance["files"]
    assert len(files) == 3
    actual = {
        str(path.relative_to(_ROOT / "src"))
        for path in (_ROOT / "src/evaluation_examples").rglob("*.py")
    }
    assert actual == set(files)
    for name, digest in files.items():
        assert hashlib.sha256((_ROOT / "src" / name).read_bytes()).hexdigest() == digest
    assert not list((_ROOT / "src/evaluation_examples").rglob("task_*.py"))
    assert "Apache License" in (_ROOT / "OSWORLD-LICENSE").read_text()


def test_census_distinguishes_runtime_type_only_and_optional_imports() -> None:
    required, optional = dependencies.import_census("""
import json
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import type_only
else:
    import runtime_else
if typing.TYPE_CHECKING:
    import another_type_only
def evaluate():
    import needed_at_evaluation
    try:
        import torch
        import lpips
    except Exception:
        import fallback
try:
    import mandatory
except ValueError:
    pass
raise RuntimeError('must not execute')
""")
    assert required == {
        "json",
        "typing",
        "runtime_else",
        "needed_at_evaluation",
        "fallback",
        "mandatory",
    }
    assert optional == {"torch", "lpips"}


def test_presence_never_imports_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / "evaluation_examples"
    package.mkdir()
    (package / "__init__.py").write_text("raise RuntimeError('parent executed')")
    (package / "helper.py").write_text("raise RuntimeError('helper executed')")
    monkeypatch.setattr(sys, "path", [str(tmp_path)])
    before = set(sys.modules)
    assert dependencies.module_present("evaluation_examples.helper")
    assert not dependencies.module_present("evaluation_examples.missing")
    assert not dependencies.module_present("missing_distribution")
    assert dependencies.module_present("os.path")
    assert set(sys.modules) == before


def test_preflight_reports_missing_without_executing_source() -> None:
    with pytest.raises(dependencies.MissingTaskDependencies, match="absent_runtime_dependency"):
        dependencies.validate_task_dependencies(
            "import absent_runtime_dependency\nraise RuntimeError('task executed')"
        )


def test_preflight_traverses_packaged_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    checked: list[str] = []

    def present(name: str) -> bool:
        checked.append(name)
        return name != "desktop_env.evaluators"

    monkeypatch.setattr(dependencies, "module_present", present)
    with pytest.raises(dependencies.MissingTaskDependencies, match="desktop_env.evaluators"):
        dependencies.validate_task_dependencies(
            "from evaluation_examples.task_class.generated_task_utils import evaluate_metric"
        )
    assert "typing" in checked


@pytest.mark.asyncio
async def test_missing_helper_stops_before_provider_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, episode_id: str
) -> None:
    from lop_osworld_v2_adapter.adapter import OSWorldV2Adapter

    from tests.unit.evaluation.adapters.osworld import fixtures
    from tests.unit.evaluation.adapters.osworld.test_secrets_and_rescue import (
        _AWS_SECRETS,
        _prepared,
        _reset,
        _workspace,
    )

    workspace = _workspace(
        tmp_path,
        {"task_plain": fixtures.PLAIN + "\nimport absent_runtime_dependency\n"},
        provider={"provider": "aws"},
    )
    adapter = OSWorldV2Adapter(workspace_root=workspace)
    await _prepared(adapter, episode_id)

    def forbidden() -> None:
        pytest.fail("provider construction reached before packaging preflight")

    monkeypatch.setattr(adapter, "_build_provider", forbidden)
    with pytest.raises(dependencies.MissingTaskDependencies, match="absent_runtime_dependency"):
        await adapter.reset_start(_reset(episode_id, "task_plain", tmp_path, _AWS_SECRETS))


def test_optional_missing_dependencies_do_not_exclude_task() -> None:
    dependencies.validate_task_dependencies(
        "try:\n import absent_optional_dependency\nexcept ImportError:\n pass"
    )
