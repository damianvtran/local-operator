"""Exercise the suite's real fixture lifecycle in reusable xdist workers.

A manually entered guard cannot prove that pytest undoes catalogue/refresher
monkeypatches before it restores the locale. Eight items on four worksteal
workers give each worker a producer and a later consumer; every item leaves
French active, so the unchanged English marker-chip assertion discriminates
without relying on the scheduler putting a particular pair together.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_locale_and_refreshers_are_restored_after_monkeypatch_teardown(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    suite = tmp_path / "test_worker_locale.py"
    suite.write_text(
        '''import pytest
from local_operator.i18n import catalogues, text
from tests.unit.tui.test_marker_chip import (
    test_the_receipt_row_is_inked_as_chrome_not_as_the_user_s_prose as receipt,
)

pytest_plugins = ["tests.conftest", "tests.unit.tui.conftest"]
real_root = catalogues.catalogue_root()
last_refresh = None


def refreshed():
    global last_refresh
    assert catalogues.catalogue_root() == real_root
    assert real_root.is_dir()
    last_refresh = text.active_locale()


text.register_refresh(refreshed)


def patched_refresher():
    raise AssertionError("locale restored before monkeypatch teardown")


@pytest.mark.parametrize("item", range(8))
async def test_worker_boundary(item, monkeypatch, tmp_path):
    # This is the exact formerly failing test, not a translated replacement.
    await receipt()
    assert last_refresh in (None, "en"), "cached constants were not refreshed"
    text.set_active_locale("fr")
    assert text.active_locale() == "fr", "the producer did not activate French"
    assert last_refresh == "fr", "the real refresh callback never ran"
    # Teardown must restore these BEFORE calling the public locale setter.
    monkeypatch.setattr(catalogues, "_CATALOGUES", tmp_path / "missing-catalogues")
    monkeypatch.setattr(text, "_refreshers", [patched_refresher])
''',
        encoding="utf-8",
    )
    home = tmp_path / "home"
    home.mkdir()
    env = {
        "HOME": str(home),
        "LOCAL_OPERATOR_CONFIG_DIR": str(home / ".local-operator"),
        "PATH": os.environ["PATH"],
        "TERM": "xterm-256color",
        "PYTHONPATH": str(repo),
    }
    result = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "run_bounded.py"),
            "--timeout",
            "180",
            "--",
            sys.executable,
            "-m",
            "pytest",
            str(suite),
            "--basetemp",
            str(tmp_path / "pytest-temp"),
            "-n",
            "4",
            "--dist",
            "worksteal",
            "--asyncio-mode=auto",
            "-q",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    print(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "8 passed" in result.stdout, result.stdout
