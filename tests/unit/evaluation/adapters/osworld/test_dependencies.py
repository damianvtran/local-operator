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


def _runtime_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    package = tmp_path / "desktop_env" / "evaluators"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("raise RuntimeError('parent executed')")
    (package / "__init__.py").write_text("raise RuntimeError('evaluators executed')")
    monkeypatch.syspath_prepend(str(tmp_path))
    return package


def test_runtime_from_imports_preserve_exports_and_reexports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _runtime_package(tmp_path, monkeypatch)
    (package / "__init__.py").write_text(
        "CONSTANT = 1\ndef function(): pass\nclass Class: pass\n"
        "from .existing import exported\nfrom . import missing_module\n"
        "raise RuntimeError('initializer executed')\n"
    )
    (package / "existing.py").write_text("exported = 1\n")
    dependencies.validate_task_dependencies(
        "from desktop_env.evaluators import CONSTANT, function, Class, exported"
    )
    with pytest.raises(dependencies.MissingTaskDependencies, match="evaluators.missing_module"):
        dependencies.validate_task_dependencies("from desktop_env.evaluators import missing_module")


def test_runtime_from_imports_keep_type_only_and_guarded_leaves_optional(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _runtime_package(tmp_path, monkeypatch)
    dependencies.validate_task_dependencies(
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n from desktop_env.evaluators import absent_type\n"
        "try:\n from desktop_env.evaluators import absent_optional\n"
        "except ImportError:\n pass\n"
    )


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

    package = _runtime_package(tmp_path, monkeypatch)
    (package / "getters").mkdir()
    (package / "getters/__init__.py").write_text("raise RuntimeError('getters executed')")
    workspace = _workspace(
        tmp_path,
        {
            "task_plain": fixtures.PLAIN
            + "\nfrom evaluation_examples.task_class.generated_task_utils import call_metric\n"
        },
        provider={"provider": "aws"},
    )
    adapter = OSWorldV2Adapter(workspace_root=workspace)
    await _prepared(adapter, episode_id)

    def forbidden() -> None:
        pytest.fail("provider construction reached before packaging preflight")

    monkeypatch.setattr(adapter, "_build_provider", forbidden)
    with pytest.raises(
        dependencies.MissingTaskDependencies, match="desktop_env.evaluators.metrics"
    ):
        await adapter.reset_start(_reset(episode_id, "task_plain", tmp_path, _AWS_SECRETS))


def test_optional_missing_dependencies_do_not_exclude_task() -> None:
    dependencies.validate_task_dependencies(
        "try:\n import absent_optional_dependency\nexcept ImportError:\n pass"
    )
