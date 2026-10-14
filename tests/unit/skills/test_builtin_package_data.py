"""The builtin skill catalog ships: package-data covers every file under it.

The catalog is read from an INSTALLED package (``skills/api.py``'s
``PACKAGED_SKILL_ROOT``), so a wheel missing it is a broken install that tests
on the source tree would never notice. This mirrors
``tests/unit/i18n/test_package_data.py`` and the mobile precedent: the
pyproject globs are read, each required glob is asserted present, every file on
disk must be covered by one — setuptools package-data globbing does not
recurse, so a new nesting level needs its own line, and this test is what fails
when it is missing — and ``importlib.resources`` is the wheel-shape proxy.
"""

from __future__ import annotations

import tomllib
from importlib.resources import files
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[3]
CATALOG_REL = "skills/builtin"
CATALOG_DIR = REPO / "local_operator" / "skills" / "builtin"

#: The globs the builtin catalog depends on. Present-or-fail, because a deleted
#: line is exactly the regression this guards.
REQUIRED_GLOBS = ("skills/builtin/*/SKILL.md", "skills/builtin/*/references/*.md")


def _package_data_globs() -> list[str]:
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    return list(data["tool"]["setuptools"]["package-data"]["local_operator"])


def _covered(rel: str, globs: list[str]) -> bool:
    """Whether a declared glob can cover a package-relative path.

    Segment-wise (``PurePath.match``), NOT ``fnmatch``: fnmatch's ``*`` crosses
    ``/`` while setuptools' glob does not, so ``skills/builtin/*/SKILL.md``
    would read as covering ``skills/builtin/a/deep/SKILL.md`` — a file that
    would still be absent from the wheel.
    """
    return any(PurePosixPath(rel).match(glob) for glob in globs)


def test_required_globs_are_declared() -> None:
    globs = _package_data_globs()
    for required in REQUIRED_GLOBS:
        assert required in globs, (
            f"pyproject package-data no longer lists {required!r}: a wheel would "
            "install without part of the builtin skill catalog"
        )


def test_segment_wise_matching_rejects_deeper_nesting() -> None:
    # The property fnmatch could not see: a DEEPER path is not covered by the
    # one-level reference glob (same as setuptools, which would leave it out of
    # the wheel while the source-tree tests passed).
    assert PurePosixPath("skills/builtin/x/SKILL.md").match("skills/builtin/*/SKILL.md")
    assert not PurePosixPath("skills/builtin/x/references/deep/y.md").match(
        "skills/builtin/*/references/*.md"
    )


def test_every_catalog_file_is_covered() -> None:
    globs = _package_data_globs()
    uncovered: list[str] = []
    for path in sorted(CATALOG_DIR.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(REPO / "local_operator").as_posix()
        if not _covered(rel, globs):
            uncovered.append(rel)
    assert not uncovered, f"builtin catalog files not covered by package-data globs: {uncovered}"


def test_only_markdown_files_ship() -> None:
    """No ``__pycache__``, no scripts, no dotfiles: Phase 1 is markdown only."""
    stray = [
        path.relative_to(CATALOG_DIR).as_posix()
        for path in sorted(CATALOG_DIR.rglob("*"))
        if path.is_file() and (path.suffix != ".md" or path.name.startswith("."))
    ]
    assert not stray, f"non-markdown files would ship in the builtin catalog: {stray}"


def test_importlib_resources_resolves_the_catalog() -> None:
    """The wheel-shape proxy: the catalog must hang off the IMPORTED package.

    A repository-relative guess can pass on the source tree while the installed
    package lacks the directory entirely; resolving through
    ``importlib.resources`` is what proves the two agree.
    """
    catalog = files("local_operator") / "skills" / "builtin"
    assert catalog.is_dir(), catalog
    assert Path(str(catalog)).resolve() == CATALOG_DIR.resolve()

    on_disk = {
        child.name
        for child in CATALOG_DIR.iterdir()
        if child.is_dir() and (child / "SKILL.md").is_file()
    }
    assert on_disk, "the catalog discovered no skill directories to check"
    packaged = {child.name for child in catalog.iterdir() if (child / "SKILL.md").is_file()}
    assert packaged == on_disk
