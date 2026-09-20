"""The bundle's guard: a utility-less stylesheet must not build green.

`local_operator/mobile/web/scripts/check-bundle.mjs` runs as npm's `postbuild`,
so it decides whether `pnpm build` succeeds — in this checkout, in CI
(mobile-web.yml), in the release (publish.yml) and in an installed tree's own
self-heal (`lop mobile install`). It exists because Tailwind emitting no
utilities is a SILENT failure: vite exits 0, the bundle serves, and the phone
renders unstyled with nothing in any log to explain it. That is precisely what
an INSTALLED tree produced (measured: 31 kB of base layer against 50 kB for
these sources, 19 class selectors against 235).

The real script is exercised here rather than a re-implementation of it: the
guard's value is that it judges a real `dist/`, and a test that models the
judgement instead of running it would keep passing if the script rotted.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
GUARD = REPO / "local_operator" / "mobile" / "web" / "scripts" / "check-bundle.mjs"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None,
    reason="node is not installed; the guard runs where the bundle is built",
)


def _run_guard(dist: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["node", str(GUARD), "--dist", str(dist)],
        capture_output=True,
        text=True,
        timeout=60,
    )


def _bundle(dist: Path, *, classes: int, stylesheet: bool = True, script: bool = True) -> None:
    """A dist/ shaped like vite's, with `classes` distinct class selectors."""
    (dist / "assets").mkdir(parents=True)
    (dist / "assets" / "app.js").write_text("console.log(1)\n", encoding="utf-8")
    link = ""
    if stylesheet:
        rules = "\n".join(f".c{index}{{color:red}}" for index in range(classes))
        (dist / "assets" / "app.css").write_text(f"{rules}\n", encoding="utf-8")
        link = '<link rel="stylesheet" href="./assets/app.css">'
    tag = '<script src="./assets/app.js"></script>' if script else ""
    (dist / "index.html").write_text(f"<html><head>{link}{tag}</head></html>", encoding="utf-8")


def test_the_guard_accepts_a_bundle_that_carries_utilities(tmp_path: Path) -> None:
    _bundle(tmp_path, classes=235)

    result = _run_guard(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "bundle ok: 235 classes" in result.stdout


def test_the_guard_rejects_a_utility_less_stylesheet(tmp_path: Path) -> None:
    """The measured degenerate shape: Tailwind's base layer and nothing else."""
    _bundle(tmp_path, classes=19)

    result = _run_guard(tmp_path)

    assert result.returncode == 1
    assert "19 classes" in result.stderr
    assert "render unstyled" in result.stderr
    assert "@source" in result.stderr


def test_the_guard_rejects_an_empty_bundle(tmp_path: Path) -> None:
    result = _run_guard(tmp_path)

    assert result.returncode == 1
    assert "produced nothing to serve" in result.stderr


def test_the_guard_rejects_an_index_with_no_stylesheet(tmp_path: Path) -> None:
    _bundle(tmp_path, classes=235, stylesheet=False)

    result = _run_guard(tmp_path)

    assert result.returncode == 1
    assert "cannot load without one of each" in result.stderr


def test_the_guard_rejects_a_dangling_asset_reference(tmp_path: Path) -> None:
    _bundle(tmp_path, classes=235)
    (tmp_path / "assets" / "app.js").unlink()

    result = _run_guard(tmp_path)

    assert result.returncode == 1
    assert "missing or empty" in result.stderr


def test_the_guard_is_wired_into_the_build(tmp_path: Path) -> None:
    """`postbuild`, not a script someone has to remember to run: an installed
    tree's self-heal drives `pnpm build` and nothing else."""
    import json

    package = json.loads(
        (REPO / "local_operator" / "mobile" / "web" / "package.json").read_text(encoding="utf-8")
    )

    assert package["scripts"]["postbuild"] == "node scripts/check-bundle.mjs"
