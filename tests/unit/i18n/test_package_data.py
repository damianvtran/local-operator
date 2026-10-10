"""The i18n catalogue and data files ship: package-data covers them.

The backend serves catalogues from an INSTALLED package and the formatter
reads the generated tables, so a wheel missing either is a broken install that
tests on the source tree would never notice. This mirrors
``tests/unit/mobile/test_web_package_data.py``: the pyproject globs are read,
each expected glob is asserted present, and every file on disk must be covered
by one — setuptools package-data globbing does not recurse, so a new
subdirectory needs a line, and this test is what fails when it is missing.
"""

from __future__ import annotations

import tomllib
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[3]
PACKAGE = REPO / "local_operator"

#: The globs the loader and formatter depend on. Present-or-fail, because a
#: deleted line is exactly the regression this guards.
REQUIRED_GLOBS = ("i18n/catalogues/*/*.json", "i18n/data/*.json")


def _package_data_globs() -> list[str]:
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    return list(data["tool"]["setuptools"]["package-data"]["local_operator"])


def _covered(rel: str, globs: list[str]) -> bool:
    """Whether a declared glob can cover a package-relative path.

    Segment-wise (`PurePath.match`), NOT `fnmatch`: fnmatch's `*` crosses `/`
    while setuptools' glob does not, so `i18n/catalogues/*/*.json` used to
    read as covering `i18n/catalogues/en/deep/x.json` — a file that would
    still be absent from the wheel (round-1 m4).
    """
    return any(PurePosixPath(rel).match(glob) for glob in globs)


def test_required_globs_are_declared() -> None:
    globs = _package_data_globs()
    for required in REQUIRED_GLOBS:
        assert required in globs, (
            f"pyproject package-data no longer lists {required!r}: a wheel would "
            "install without its catalogues or generated tables"
        )


def test_segment_wise_matching_rejects_deeper_nesting() -> None:
    # The property fnmatch could not see: a DEEPER path is not covered by a
    # two-level glob (same as setuptools, which would leave it out of the
    # wheel while the source-tree tests passed).
    assert PurePosixPath("i18n/catalogues/en/x.json").match("i18n/catalogues/*/*.json")
    assert not PurePosixPath("i18n/catalogues/en/deep/x.json").match("i18n/catalogues/*/*.json")
    assert not PurePosixPath("i18n/data/deep/x.json").match("i18n/data/*.json")


def test_every_i18n_data_file_is_covered() -> None:
    globs = _package_data_globs()
    uncovered: list[str] = []
    for base in ("i18n/catalogues", "i18n/data"):
        for path in sorted((PACKAGE / base).rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(PACKAGE).as_posix()
            if not _covered(rel, globs):
                uncovered.append(rel)
    assert not uncovered, f"i18n files not covered by package-data globs: {uncovered}"
