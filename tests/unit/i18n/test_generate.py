"""The generator: per-key function shape, shipped-set computation, drift."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from local_operator.i18n import catalogues
from local_operator.i18n import format as fmt

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture()
def generated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, i18n_generate):
    """A fixture catalogue root + emitted tables wired into the generator."""
    root = tmp_path / "catalogues"
    (root / "en").mkdir(parents=True)
    emitted = tmp_path / "emitted"
    emitted.mkdir()
    data = Path(fmt.__file__).parent / "data"
    for name in ("plural_rules.json", "formats.json"):
        shutil.copy(data / name, emitted / name)
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    monkeypatch.setattr(i18n_generate, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(i18n_generate, "KEYS_DIR", tmp_path / "keys")
    monkeypatch.setattr(i18n_generate, "LEDGER", tmp_path / "ledger.json")
    return root, emitted


def test_per_key_functions_are_emitted_with_typed_kwargs(generated, i18n_generate) -> None:
    root, emitted = generated
    (root / "en" / "demo.words.json").write_text(
        json.dumps(
            {
                "demo.words.hello": "Hello, {name}!",
                "demo.words.files": "{count, plural, one {# file} other {# files}}",
                "demo.words.updated": "Updated {when, date, short}",
                "demo.words.plain": "plain text, no params",
                "demo.words.import": "keyword collision",
            }
        ),
        encoding="utf-8",
    )
    planned = i18n_generate._planned_files(emitted)
    module = planned[Path(i18n_generate.KEYS_DIR) / "demo_words.py"]
    expected_body = """\


def files(*, count: int | float) -> Msg:
    return Msg("demo.words.files", {"count": count})


def hello(*, name: str) -> Msg:
    return Msg("demo.words.hello", {"name": name})


def import_() -> Msg:
    return Msg("demo.words.import", {})


def plain() -> Msg:
    return Msg("demo.words.plain", {})


def updated(*, when: datetime) -> Msg:
    return Msg("demo.words.updated", {"when": when})
"""
    assert module == i18n_generate.GENERATED_HEADER + (
        "from datetime import datetime\n\nfrom ..messages import Msg\n\n\n"
    ) + expected_body.lstrip("\n")
    # The generated module is what black/isort would leave alone.
    check = subprocess.run(
        [sys.executable, "-m", "black", "--check", "-"],
        input=module,
        capture_output=True,
        text=True,
        check=False,
    )
    if check.returncode != 0 and "No module named" not in check.stderr:
        pytest.fail(f"generated module is not black-clean:\n{check.stdout}\n{check.stderr}")


def test_empty_namespace_module_has_no_dangling_import(generated, i18n_generate) -> None:
    root, emitted = generated
    (root / "en" / "wire.empty.json").write_text("{}", encoding="utf-8")
    planned = i18n_generate._planned_files(emitted)
    module = planned[Path(i18n_generate.KEYS_DIR) / "wire_empty.py"]
    assert module == i18n_generate.GENERATED_HEADER.rstrip() + "\n"
    assert "from ..messages import Msg" not in module
    assert "from datetime" not in module


def test_shipped_set_requires_fresh_passed_entries(generated, i18n_generate) -> None:
    root, emitted = generated
    demo = {"demo.x": "x"}
    (root / "en" / "demo.json").write_text(json.dumps(demo), encoding="utf-8")
    fresh = catalogues.catalogue_sha256("en", "demo")
    ledger = {
        "schema": 1,
        "entries": [
            {"locale": "fr", "namespace": "demo", "status": "passed", "source_sha256": fresh},
            {"locale": "es", "namespace": "demo", "status": "passed", "source_sha256": "stale"},
            {"locale": "ru", "namespace": "demo", "status": "failed", "source_sha256": fresh},
            {"locale": "vi", "namespace": "other", "status": "passed", "source_sha256": "x"},
        ],
    }
    Path(i18n_generate.LEDGER).write_text(json.dumps(ledger), encoding="utf-8")
    assert i18n_generate._shipped_locales() == ["en", "fr"]
    # No ledger at all: en alone, not an empty set.
    Path(i18n_generate.LEDGER).unlink()
    assert i18n_generate._shipped_locales() == ["en"]


def test_drift_check_passes_on_the_committed_tree() -> None:
    if shutil.which("node") is None:
        pytest.skip("node is required for the emitter; CI's i18n job covers it")
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "i18n" / "generate.py"), "--check"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "byte-identical" in result.stdout
