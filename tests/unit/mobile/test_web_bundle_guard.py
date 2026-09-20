"""The bundle's guard: it must fail in BOTH directions, and not silently.

`local_operator/mobile/web/scripts/check-bundle.mjs` runs as npm's `postbuild`,
so it decides whether `pnpm build` succeeds — in this checkout, in CI
(mobile-web.yml), in the release (publish.yml) and in an installed tree's own
self-heal (`lop mobile install`). It exists because Tailwind emitting the wrong
stylesheet is a SILENT failure: vite exits 0, the bundle serves, and the phone
renders unstyled or half-styled with nothing in any log to explain it.

Both directions are measured on this defect family:

* UNDER-inclusion — a stylesheet missing classes the app renders with. The
  installed tree produced 62 classes against 235 for these sources, and the flat
  "at least 100 classes" floor an earlier revision used passed a 96 kB / 852-class
  POLLUTED bundle, which is why the check is per token.
* OVER-inclusion — a scan that found a tree the build does not own
  (node_modules), which brings every real class along with hundreds of
  strangers.

The real script is exercised here rather than a re-implementation of it: the
guard's value is that it judges a real `dist/` against real sources, and a test
that models the judgement instead of running it would keep passing if the
script rotted.
"""

from __future__ import annotations

import json
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


def _run_guard(web: Path, dist: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["node", str(GUARD), "--web", str(web), "--dist", str(dist)],
        capture_output=True,
        text=True,
        timeout=60,
    )


def _tree(
    tmp_path: Path,
    *,
    tokens: tuple[str, ...] = ("flex", "text-loud"),
    css: str | None = None,
    source: str = '<span className="flex text-loud" />',
    stylesheet: bool = True,
    script: bool = True,
) -> tuple[Path, Path]:
    """A web tree plus its built `dist/`, wired the way vite writes them."""
    web = tmp_path / "web"
    (web / "src").mkdir(parents=True)
    (web / "index.html").write_text('<html><body><div id="root"></div></body></html>')
    (web / "src" / "app.tsx").write_text(
        f"export const App = () => ({source});\n", encoding="utf-8"
    )
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "assets" / "app.js").write_text("console.log(1)\n", encoding="utf-8")
    link = ""
    if stylesheet:
        rules = css if css is not None else "\n".join(f".{token}{{color:red}}" for token in tokens)
        (dist / "assets" / "app.css").write_text(f"{rules}\n", encoding="utf-8")
        link = '<link rel="stylesheet" href="./assets/app.css">'
    tag = '<script src="./assets/app.js"></script>' if script else ""
    (dist / "index.html").write_text(f"<html><head>{link}{tag}</head></html>", encoding="utf-8")
    return web, dist


def test_the_guard_accepts_a_bundle_that_covers_its_sources(tmp_path: Path) -> None:
    web, dist = _tree(tmp_path)

    result = _run_guard(web, dist)

    assert result.returncode == 0, result.stderr
    assert "bundle ok: 2 classes for 2 tokens" in result.stdout


def test_the_guard_rejects_a_bundle_missing_a_class_the_app_renders(
    tmp_path: Path,
) -> None:
    """The under-inclusion direction, and the shape of the real defect: the
    stylesheet is there, loads, exits 0 — and one class the app names has no
    rule, so that element renders unstyled."""
    web, dist = _tree(tmp_path, css=".flex{color:red}")

    result = _run_guard(web, dist)

    assert result.returncode == 1
    assert "no rule for 1 class(es)" in result.stderr
    assert "text-loud" in result.stderr


def test_the_guard_rejects_a_polluted_scan(tmp_path: Path) -> None:
    """The over-inclusion direction: a scan that wandered into node_modules has
    every real class AND hundreds of strangers, so a per-token check alone (and
    any flat floor) passes it. Measured shapes of this: 695 and 852 selectors
    for these 225 sources."""
    noise = "\n".join(f".n{index}{{color:red}}" for index in range(400))
    web, dist = _tree(tmp_path, css=f".flex{{color:red}}\n.text-loud{{color:red}}\n{noise}")

    result = _run_guard(web, dist)

    assert result.returncode == 1
    assert "the scan found a tree the build does not own" in result.stderr


def test_the_guard_accepts_a_documented_hook_class(tmp_path: Path) -> None:
    """`lo-loadbar` carries no rule of its own (its animation lives on a child
    rule), so the check has to tolerate the one argued exception — and only
    that one."""
    web, dist = _tree(
        tmp_path,
        tokens=("flex",),
        css=".flex{color:red}",
        source='<span className="flex lo-loadbar" />',
    )

    result = _run_guard(web, dist)

    assert result.returncode == 0, result.stderr


def test_the_guard_rejects_an_unnamed_class(tmp_path: Path) -> None:
    """The other half of the exception: a class nobody argued for still fails."""
    web, dist = _tree(
        tmp_path,
        tokens=("flex",),
        css=".flex{color:red}",
        source='<span className="flex lo-mystery" />',
    )

    result = _run_guard(web, dist)

    assert result.returncode == 1
    assert "lo-mystery" in result.stderr


def test_the_guard_ignores_comments_and_comparisons(tmp_path: Path) -> None:
    """Class-shaped text that is not a class: a token in a comment inside a
    `className` expression, and a status value compared against (measured: the
    real sources compare `state === "loading"`, `t.status === "done"` and
    friends, which are values, not utilities)."""
    web, dist = _tree(
        tmp_path,
        tokens=("flex",),
        css=".flex{color:red}",
        source=('<span className={cn("flex", state === "loading" && /* zzz-hidden */ "flex")} />'),
    )

    result = _run_guard(web, dist)

    assert result.returncode == 0, result.stderr


def test_the_guard_rejects_an_empty_bundle(tmp_path: Path) -> None:
    web = tmp_path / "web"
    (web / "src").mkdir(parents=True)
    dist = tmp_path / "dist"
    dist.mkdir()

    result = _run_guard(web, dist)

    assert result.returncode == 1
    assert "produced nothing to serve" in result.stderr


def test_the_guard_rejects_an_index_with_no_stylesheet(tmp_path: Path) -> None:
    web, dist = _tree(tmp_path, stylesheet=False)

    result = _run_guard(web, dist)

    assert result.returncode == 1
    assert "cannot load without one of each" in result.stderr


def test_the_guard_rejects_a_dangling_asset_reference(tmp_path: Path) -> None:
    web, dist = _tree(tmp_path)
    (dist / "assets" / "app.js").unlink()

    result = _run_guard(web, dist)

    assert result.returncode == 1
    assert "missing or empty" in result.stderr


def test_the_guard_refuses_to_pass_vacuously(tmp_path: Path) -> None:
    """No sources is not a pass: a renamed or absent `src/` would otherwise make
    every direction above vacuous."""
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "assets" / "app.js").write_text("1\n", encoding="utf-8")
    (dist / "assets" / "app.css").write_text(".flex{color:red}\n", encoding="utf-8")
    (dist / "index.html").write_text(
        '<html><head><link rel="stylesheet" href="./assets/app.css">'
        '<script src="./assets/app.js"></script></head></html>',
        encoding="utf-8",
    )

    result = _run_guard(tmp_path / "web", dist)

    assert result.returncode == 1
    assert "no class tokens found" in result.stderr


def test_the_guard_is_wired_into_the_build() -> None:
    """`postbuild`, not a script someone has to remember to run: an installed
    tree's self-heal drives `pnpm build` and nothing else."""
    package = json.loads(
        (REPO / "local_operator" / "mobile" / "web" / "package.json").read_text(encoding="utf-8")
    )

    assert package["scripts"]["postbuild"] == "node scripts/check-bundle.mjs"


def test_the_guard_passes_the_published_bundle_for_these_sources() -> None:
    """The shipped tree is the reference the check has to accept: if the guard
    rejects a bundle that Tailwind built from these sources, it is wrong, not
    the bundle."""
    web = REPO / "local_operator" / "mobile" / "web"
    dist = web / "dist"
    if not (dist / "index.html").exists():
        pytest.skip("no local dist/ (the bundle is built by pnpm build, not by this suite)")

    result = subprocess.run(
        ["node", str(GUARD), "--web", str(web), "--dist", str(dist)],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
