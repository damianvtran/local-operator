"""Developer CLI failures must be explicit, not successful-looking evidence."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.visual_capture import CaptureProfile, font_provenance

ROOT = Path(__file__).resolve().parents[3]


def test_unavailable_font_query_is_explicit(monkeypatch) -> None:
    monkeypatch.setattr("scripts.visual_capture.shutil.which", lambda _: None)
    font_provenance.cache_clear()
    report = font_provenance(CaptureProfile())
    assert report["status"].startswith("unresolved")
    font_provenance.cache_clear()


@pytest.mark.parametrize("argument", ["0x0", "-1x20", "100x9999", "oops", "20xnan"])
def test_page_cli_rejects_invalid_grid(argument: str, tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/pages_shot.py"),
            str(tmp_path / "out.svg"),
            "welcome",
            argument,
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert not (tmp_path / "out.svg").exists()


def test_gallery_inventory_lists_and_rejects_unknown_case(tmp_path: Path) -> None:
    command = [sys.executable, str(ROOT / "scripts/visual_gallery.py")]
    listed = subprocess.run(command + ["--list"], capture_output=True, text=True)
    assert listed.returncode == 0
    assert "reference-welcome-158x44" in listed.stdout
    assert "page-usage-error" in listed.stdout
    failed = subprocess.run(
        command + [str(tmp_path), "--case", "does-not-exist"], capture_output=True, text=True
    )
    assert failed.returncode == 2
    assert "unknown case" in failed.stderr


def test_bootstrap_probe_requires_choice_and_isolates_real_boot(tmp_path: Path) -> None:
    command = [sys.executable, str(ROOT / "scripts/eager_boot_shot.py")]
    output = tmp_path / "boot.svg"
    sentinel = tmp_path / "operator-config" / "keep.txt"
    sentinel.parent.mkdir()
    sentinel.write_text("do not modify")
    env = dict(os.environ, HOME=str(tmp_path), LOCAL_OPERATOR_CONFIG_DIR=str(sentinel.parent))
    rejected = subprocess.run(command + [str(output)], env=env, capture_output=True, text=True)
    assert rejected.returncode == 2
    assert not output.exists()
    result = subprocess.run(
        command + [str(output), "--isolated"], env=env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert "is_cold:          True" in result.stdout
    assert "effective model:  unconfigured" in result.stdout
    assert "mcp servers:      0" in result.stdout
    geometry = json.loads(output.with_suffix(".geometry.json").read_text())
    assert geometry["native_pixels"] == [960, 544]
    assert sentinel.read_text() == "do not modify"


def test_all_sample_isolation_precedes_app_imports() -> None:
    import ast

    for path in (ROOT / "scripts").glob("*.py"):
        text = path.read_text()
        if "save_capture(app," not in text or path.name == "visual_capture.py":
            continue
        if path.name == "eager_boot_shot.py":
            # Explicit live bootstrap is the sole opt-in exception; the default
            # gallery uses its isolated unconfigured path, never operator auth.
            assert "CAPTURE_REQUIRES_LIVE_OPT_IN = True" in text
        tree = ast.parse(text)
        isolation = next(
            n.lineno
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "isolate_capture"
        )
        imports = [
            n.lineno
            for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("local_operator")
        ]
        assert isolation < min(imports), path.name


def test_isolation_ignores_ambient_config_and_preserves_sentinel(tmp_path: Path) -> None:
    sentinel = tmp_path / "config" / "keep.txt"
    sentinel.parent.mkdir()
    sentinel.write_text("do not modify")
    code = """
import json, os
from pathlib import Path
from scripts.visual_capture import isolate_capture
isolate_capture()
print(json.dumps({'home': os.environ['HOME'], 'config': os.environ['LOCAL_OPERATOR_CONFIG_DIR']}))
"""
    env = dict(os.environ, HOME=str(tmp_path), LOCAL_OPERATOR_CONFIG_DIR=str(sentinel.parent))
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True
    )
    roots = json.loads(completed.stdout)
    assert roots["home"] != str(tmp_path)
    assert roots["config"].startswith(roots["home"])
    assert not Path(roots["home"]).exists(), "temporary capture HOME leaked after exit"
    assert sentinel.read_text() == "do not modify"
