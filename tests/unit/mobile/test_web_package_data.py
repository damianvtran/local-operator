"""The mobile web tree must ship every file the installed tree builds from.

``dist/`` is gitignored, so a source-snapshot install — what ``lop-update``
puts in a generation tree — has the web sources and no bundle, which leaves
``lop mobile install`` rebuilding them in place as the ONE documented repair
for the 503 the daemon then serves. That rebuild runs ``pnpm build``, whose
first step is ``tsc -b`` over ``tsconfig.app.json`` with ``include: ["src"]``:
the installed tree compiles EVERY file under ``src`` — the ``*.test.tsx`` files
included, since they ship too — so anything those files read is a build input.

``mobile/web/src/fixtures/*.json`` was left out of
``[tool.setuptools.package-data]``, and the shipped tree therefore could not
build at all: ``src/model-sheet.order.test.tsx`` imports
``./fixtures/models.ranked.json``, so ``tsc`` failed with TS2307 before vite
ran, on every ``lop-update`` install — and the documented repair failed the
same way, so the phone stayed on 503 "bundle not built" with no way back.
The same audit found ``src/lib/format.parity.json``, which
``src/lib/format.test.ts`` reads with ``readFileSync``: no import for a scanner
to follow, so only the whole-tree check below can see it.

The checks below are DERIVED from the ``src/`` tree and the pyproject globs
rather than naming a file, because the defect is an omission and a hand-written
list is exactly what an omission slips past. A new source directory, a new
fixture type, a new asset import, or a glob that mistypes a directory fails
here — in the unit gate — instead of on someone's phone.

``setuptools`` is not a test dependency, so the glob→path matching is modelled:
``package-data`` globs are matched against the path relative to the package
directory, and ``*`` does not cross a ``/``, which is the per-directory shape
the globs in ``pyproject.toml`` are written to (see the note beside them).
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PACKAGE_DIR = REPO / "local_operator"
PYPROJECT = REPO / "pyproject.toml"
WEB = PACKAGE_DIR / "mobile" / "web"
WEB_SRC = WEB / "src"

#: What the web tree holds but the wheel does not carry, mirroring the ignore
#: rules in ``mobile/web/.gitignore`` (plus pnpm's in-tree store): build output,
#: installed dependencies and tsc's cache. The rule this test enforces is "git
#: would track it, so an installed tree needs it"; anything else reaching this
#: list means the exclusions here and that file have drifted apart.
NOT_SHIPPED = ("dist/", "node_modules/", ".pnpm-store/", "*.tsbuildinfo")

#: The extensions a bundler consumes as source rather than as data. Everything
#: else a source names is an asset, and an asset has to ship as package data.
SOURCE_SUFFIXES = {".ts", ".tsx", ".css"}

#: `import x from "./y"`, `import "./y.css"`, `import type … from "./y"`,
#: `export … from "./y"` and `import("./y")` — the shapes a relative import of a
#: sibling takes here. The spec must start with a dot, which is what keeps
#: package imports (`from "react"`) out of the scan.
IMPORT_RE = re.compile(r"""(?:from|import)\s*\(?\s*['"](?P<spec>\.[^'"]*)['"]""")

#: `@import "./themes.generated.css"` — a stylesheet's own relative import. The
#: phone's `src/styles/index.css` reaches the generated theme sheet that way,
#: and Tailwind's bare `@import "tailwindcss"` is not a path, so the dot is
#: required here too.
CSS_IMPORT_RE = re.compile(r"""@import\s+(?:url\(\s*)?['"](?P<spec>\.[^'"]*)['"]""")


def _package_data_globs() -> list[str]:
    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    globs = config["tool"]["setuptools"]["package-data"]["local_operator"]
    assert isinstance(globs, list), "package-data is not a list of globs"
    return [glob for glob in globs if isinstance(glob, str)]


def _glob_regex(pattern: str) -> re.Pattern[str]:
    """One package-data glob as a regex anchored at the package directory.

    ``*`` stops at a separator, which is how setuptools' data-file globs
    behave and why the source tree is enumerated one directory at a time
    instead of with ``**``.
    """
    out: list[str] = []
    for char in pattern:
        if char == "*":
            out.append("[^/]*")
        elif char == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(char))
    return re.compile("".join(out) + r"\Z")


def _shipped(relative: str, globs: list[str]) -> bool:
    return any(_glob_regex(glob).fullmatch(relative) for glob in globs)


def _tree_files() -> list[str]:
    """Every file the wheel must carry from the web tree, package-relative."""
    ignored = tuple(_glob_regex(pattern) for pattern in NOT_SHIPPED)
    files: list[str] = []
    for path in WEB.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(WEB).as_posix()
        ancestors = [relative, *_directories(relative)]
        if any(pattern.fullmatch(name) for pattern in ignored for name in ancestors):
            continue
        files.append(path.relative_to(PACKAGE_DIR).as_posix())
    return sorted(files)


def _directories(relative: str) -> list[str]:
    """``a/b/c.txt`` -> ``a/b/``, ``a/`` — the prefixes an ignore rule matches."""
    parts = relative.split("/")[:-1]
    return ["/".join(parts[: index + 1]) + "/" for index in range(len(parts))]


def _resolve(importer: Path, spec: str) -> Path | None:
    """The file a relative import names, or None when nothing resolves.

    Probing is decided by whether the spec AS WRITTEN is a file, never by
    ``Path(spec).suffix``: this tree ships ``src/projects-status.generated.ts``,
    whose import spec is ``./projects-status.generated``, and ``suffix`` reads
    that last dot as ``.generated`` — so a suffix-gated probe reported a file
    that IS in the tree as missing (review round 7, Q2). The relaxation still
    resolves nothing that is not there: every probed candidate is checked with
    ``is_file()``, and a spec with no file at all — with or without a dot in
    its name — comes back None, which is what the asset check turns into a
    failure."""
    base = importer.parent / spec
    if base.is_file():
        return base.resolve()
    candidates = [
        *(base.with_name(base.name + suffix) for suffix in (".ts", ".tsx", ".css")),
        base / "index.ts",
        base / "index.tsx",
    ]
    # `resolve()` collapses the `../` a sibling import carries, so the path that
    # comes back is the one package-data globs are matched against.
    return next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)


def _imported_specs() -> list[tuple[str, str]]:
    """``(importer, spec)`` for every relative import in the shipped sources."""
    found: list[tuple[str, str]] = []
    for path in sorted(WEB_SRC.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in SOURCE_SUFFIXES:
            continue
        pattern = CSS_IMPORT_RE if path.suffix == ".css" else IMPORT_RE
        text = path.read_text(encoding="utf-8")
        found += [
            (path.relative_to(WEB_SRC).as_posix(), match.group("spec"))
            for match in pattern.finditer(text)
        ]
    return found


def test_every_file_in_the_web_tree_ships() -> None:
    globs = _package_data_globs()
    unshipped = [relative for relative in _tree_files() if not _shipped(relative, globs)]
    assert not unshipped, (
        "the installed tree builds from its own copy of the web tree, so a file left out of "
        "[tool.setuptools.package-data] is a file that tree does not have — which is how a "
        "missing JSON fixture made `tsc -b` fail, and how a missing .gitignore made Tailwind "
        "emit a utility-less stylesheet at exit 0. Add a glob for: " + ", ".join(unshipped)
    )


def test_every_asset_the_sources_import_is_shipped() -> None:
    globs = _package_data_globs()
    problems: list[str] = []
    for importer, spec in _imported_specs():
        resolved = _resolve(WEB_SRC / importer, spec)
        if resolved is None:
            problems.append(f"{importer} imports {spec!r}, which is not in the tree")
            continue
        try:
            relative = resolved.relative_to(PACKAGE_DIR).as_posix()
        except ValueError:
            problems.append(f"{importer} imports {spec!r}, which is outside the package")
            continue
        if not _shipped(relative, globs):
            problems.append(f"{importer} imports {spec!r}: {relative} matches no package-data glob")
    assert (
        not problems
    ), "an import the installed tree resolves has to resolve there too. Uncovered: " + "; ".join(
        problems
    )


def test_the_relaxed_resolver_still_reports_a_missing_asset(tmp_path: Path) -> None:
    """`_resolve` is SHARED by both asset checks, so the relaxation for a
    dotted basename is proven in BOTH directions: the shape it exists for
    resolves, and everything genuinely absent still comes back None. A resolver
    that answered for files that are not there would make the asset check a
    formality — which is the failure mode a relaxed helper invites."""
    (tmp_path / "present.ts").write_text("", encoding="utf-8")
    (tmp_path / "present.generated.ts").write_text("", encoding="utf-8")
    importer = tmp_path / "importer.tsx"

    assert _resolve(importer, "./present") is not None
    assert _resolve(importer, "./present.generated") is not None
    assert _resolve(importer, "./absent") is None
    assert _resolve(importer, "./absent.generated") is None
    assert _resolve(importer, "./absent.json") is None
    assert _resolve(importer, "../outside/either") is None

    # The real tree, in the shape the relaxation was added for, and the same
    # shape when it names nothing.
    status = WEB_SRC / "projects-status.test.tsx"
    assert _resolve(status, "./projects-status.generated") is not None
    assert _resolve(status, "./projects-status.absent") is None


def test_the_shipping_checks_are_sensitive_to_the_files_they_watch() -> None:
    """A check is only as good as its ability to fail. Pin that the glob match
    rejects a name no glob covers, and that the `.d.mts` this round added is
    the glob — not the assertion — doing the work."""
    globs = _package_data_globs()
    # Package-relative: the globs are matched against the path BELOW
    # `local_operator/`, which is the shape `_tree_files()` and the asset check
    # both feed `_shipped`.
    declaration = "mobile/web/scripts/generate-projects-status.d.mts"
    assert _shipped("mobile/web/scripts/generate-projects-status.mjs", globs)
    assert _shipped(declaration, globs)
    assert not _shipped("mobile/web/scripts/build.log", globs)
    assert not _shipped("local_operator/mobile/web/scripts/generate-projects-status.mjs", globs)
    # Remove the glob this round added and the check must catch the file again.
    assert not _shipped(declaration, [glob for glob in globs if not glob.endswith(".d.mts")])


def test_the_checks_still_see_the_shape_that_broke() -> None:
    """Both checks derive from the tree, so a moved tree would make them pass
    by finding nothing. Pin that a source still imports a data asset."""
    assets = [
        (importer, spec)
        for importer, spec in _imported_specs()
        if Path(spec).suffix not in SOURCE_SUFFIXES
    ]
    assert assets, (
        "no shipped source imports a non-source asset any more; if the fixtures moved, "
        "these checks are now vacuous and need to follow them"
    )


def _gitignore_patterns(path: Path) -> list[str]:
    """The active patterns in a .gitignore, in file order."""
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return lines


def test_the_web_tree_mirrors_every_root_ignore_rule_it_can_defeat() -> None:
    """`!src/**/` un-ignores src's DIRECTORIES — the scanner needs the walk —
    and a directory rule is exactly what that defeats: without the mirror at the
    bottom of `mobile/web/.gitignore`, `mobile/web/src/build/` becomes visible
    again. The mirror is defined mechanically (every unanchored, non-negated
    root pattern), so this test is a comparison rather than a maintained list:
    add a rule to the root and this fails until the mirror matches."""
    root = _gitignore_patterns(REPO / ".gitignore")
    mirrored = set(_gitignore_patterns(WEB / ".gitignore"))
    # An unanchored pattern matches at any depth; a leading `/` or `!` changes
    # the meaning, and both are excluded here (the mirror is of the plain ones).
    unanchored = [p for p in root if not p.startswith(("/", "!"))]
    missing = [p for p in unanchored if p not in mirrored]

    assert not missing, (
        "mobile/web/.gitignore re-includes directories under src/ for Tailwind's scanner, so "
        "every unanchored rule from the repository root has to be restated after that (last "
        "match wins), or a path the root ignores becomes visible here. Missing: "
        + ", ".join(missing)
    )
