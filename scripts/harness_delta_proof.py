#!/usr/bin/env python3
"""Prove a post-clean-round delta touches only test harness — or open a round.

Why this exists: local-operator-mobile PR #156 ran 40 numbered agent-review
rounds for a product change that was clean at round 3 (2026-11-03); the last
product-code commit landed 2026-11-05 02:14 and 20 failed native attempts plus
41.4 h of wall clock followed it. The rounds after round 3 reviewed evidence
harness (CI probes, fixtures, mock relays), not product. This script lets a
harness-only delta skip the numbered round — and is built so it cannot launder
a product change (``guide://delivery``, "Harness-only delta").

The invariant the exemption depends on, and the only thing this proves:

    No byte the shipping pipeline consumes differs between REVIEWED and HEAD.

HARNESS-ONLY (exit 0) requires ALL of:

1. Same base: ``merge-base(base, reviewed) == merge-base(base, head)``. A
   merged or rebased base puts upstream bytes into ``reviewed..head``; that is
   ``scripts/rebase_proof.py``'s job, never this one's.
2. The manifest ``.github/harness-paths`` exists at head and is unchanged in
   ``reviewed..head``. The manifest is a product path: the exemption set cannot
   grow without a reviewed round. ``--manifest`` substitutes an external file
   for retrospective replays (labelled ``external (replay)``); then nothing
   in-repo is checked for condition 2.
3. Every changed path — added, modified, deleted, BOTH sides of a rename
   (``--no-renames``), and mode-only changes — classifies as ``harness``.
4. No changed path is ship-reachable. The ship closure is every file at head
   matching a ``ship:`` glob plus, transitively, every file a closure file
   NAMES: its repo path, its basename with extension, its dotted Python module,
   a directory spelled as one (``scripts/ci/``, trailing slash), or a relative
   import/require/dynamic-import/``source``/Python-relative specifier that
   resolves to it (trying .ts .tsx .js .mjs .cjs .json .py .sh and /index.*).
   Named product files join the closure too, not only harness files: a ship
   workflow can run an undeclared product script that sources a harness file.

Classification: a path matching both a ``harness:`` and a ``ship:`` glob takes
the MORE SPECIFIC glob (longer literal prefix before the first wildcard; ties go
to ship). That is what lets mobile keep ``harness: src/testing/**`` inside
``ship: src/**`` while local-operator keeps ``ship: .github/workflows/ci.yml``
inside ``harness: .github/workflows/**``. A path matching neither is product.

Conservative by design: name matching is lexical, so a false positive opens a
round (the cost the gate already pays) while a false negative would ship an
unreviewed byte. Known lexical limits, each measured on this repo: a bare
directory word (``e2e``, ``tests/unit``) never counts — expanding those made
522 of 522 harness files "reachable" through unrelated English mentions and the
exemption inert; prose files (.md/.txt/.rst) contribute only their exact-path
and specifier hits; and a path assembled at runtime from fragments no file
spells out cannot be seen, so declare such files under ``ship:``.
Exit 0 = HARNESS-ONLY, 1 = ROUND REQUIRED (each reason named), 2 = could
not be computed (bad ref, git failure, no manifest) — a missing proof is
never a pass.

``--split`` adds the measured split trigger: added lines under harness vs
everything else since ``merge-base(base, head)``; harness > product → SPLIT
(PR #156 crossed at round 1, 2,787 vs 2,764, and stayed crossed from round 11).

Run (from the repository, after fetching both heads and the base):
    python scripts/harness_delta_proof.py <reviewed> <head> [--base origin/main]
        [--repo PATH] [--manifest PATH] [--split]
"""

from __future__ import annotations

import argparse
import functools
import posixpath
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

MANIFEST_PATH = ".github/harness-paths"
#: Extensions tried for an extensionless relative specifier, in resolver order.
_RESOLVE_EXTS = (".ts", ".tsx", ".js", ".mjs", ".cjs", ".json", ".py", ".sh")
_PROSE_EXTS = {".md", ".txt", ".rst"}
#: Path-like words: the characters a repo path, basename or dotted module uses.
#: Everything else (quotes, ``${}``, ``:``, spaces, ``*``) separates tokens, so
#: ``${root}testing/cookie-jar-fetch.ts`` yields ``testing/cookie-jar-fetch.ts``.
_TOKEN = re.compile(r"[A-Za-z0-9_.@/+-]+")
#: JS/TS string specifiers starting ``./`` or ``../`` — static import/export
#: from, require(), dynamic import() — in any quote style. Template literals
#: with ``${...}`` interpolations also match, which is how PR #156's fixture
#: reference ``${root}testing/cookie-jar-fetch.ts`` counts as naming.
_JS_SPEC = re.compile(r"""['"`](\.\.?/[^'"`\s]*)['"`]""")
#: ``source x`` / ``. x`` shell include. LINEARITY IS A CONSTRAINT: no
#: quantified class may straddle a newline — a failing match re-scans from
#: every line start, so a greedy multi-line class costs O(n^2) (measured:
#: 36 min of CPU on one 120 KB module before this shape).
_SH_SOURCE = re.compile(r"""(?m)(?:^|[;&|(])[ \t]*(?:source|\.)[ \t]+["']?([^\s;|&)'\"]+)""")
#: Python relative import: ``from ..pkg.mod import a, b``. Same linearity
#: rule: the imported-name run stops at end of line, never crossing one.
_PY_RELATIVE = re.compile(r"""(?m)^[ \t]*from[ \t]+(\.+)([\w.]*)[ \t]+import[ \t]+([^\n]*)""")


class ProofError(RuntimeError):
    """The proof could not be computed; the caller exits 2, never 0."""


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise ProofError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


# --- manifest -----------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _glob_regex(glob: str) -> re.Pattern[str]:
    """git-pathspec-like, anchored at the repo root: ``**`` any depth (zero or
    more segments), ``*``/``?`` within one segment."""
    out, i = [], 0
    while i < len(glob):
        if glob.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif glob.startswith("**", i):
            out.append(".*")
            i += 2
        elif glob[i] == "*":
            out.append("[^/]*")
            i += 1
        elif glob[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(glob[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


@dataclass
class Manifest:
    harness: list[str] = field(default_factory=list)
    ship: list[str] = field(default_factory=list)
    source: str = ""

    @classmethod
    def parse(cls, text: str, source: str) -> Manifest:
        manifest = cls(source=source)
        for number, raw in enumerate(text.splitlines(), 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            kind, sep, glob = line.partition(":")
            if not sep or kind.strip() not in ("harness", "ship") or not glob.strip():
                raise ProofError(
                    f"manifest line {number} is not 'harness: <glob>' or 'ship: <glob>'"
                )
            getattr(manifest, kind.strip()).append(glob.strip())
        return manifest

    def classify(self, path: str) -> str:
        """``ship`` | ``harness`` | ``product`` — most specific matching glob."""

        def best(globs: list[str]) -> int:
            hits = [
                len(re.split(r"[*?]", g, maxsplit=1)[0])
                for g in globs
                if _glob_regex(g).match(path)
            ]
            return max(hits, default=-1)

        ship, harness = best(self.ship), best(self.harness)
        if ship < 0 and harness < 0:
            return "product"
        return "ship" if ship >= harness else "harness"


# --- ship closure ---------------------------------------------------------------


def _blobs(repo: Path, rev: str, paths: list[str]) -> dict[str, str]:
    """Read many blobs in one ``git cat-file --batch`` (latin-1: lexical scan
    only, so binary content is searched rather than skipped)."""
    if not paths:
        return {}
    request = "".join(f"{rev}:{p}\n" for p in paths).encode()
    proc = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "--batch"],
        input=request,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise ProofError(f"git cat-file --batch failed: {proc.stderr.decode().strip()}")
    out, data, pos = {}, proc.stdout, 0
    for path in paths:
        end = data.index(b"\n", pos)
        header = data[pos:end].split()
        pos = end + 1
        if len(header) < 3 or header[1] != b"blob":  # submodule/missing: nothing to read
            continue
        size = int(header[2])
        out[path] = data[pos : pos + size].decode("latin-1")
        pos += size + 1
    return out


class _Index:
    """Lookup tables from a name a file can carry to the head paths it denotes."""

    def __init__(self, paths: list[str]) -> None:
        self.paths = set(paths)
        self.by_basename: dict[str, set[str]] = {}
        self.by_module: dict[str, set[str]] = {}
        self.by_dir: dict[str, set[str]] = {}
        for p in paths:
            self.by_basename.setdefault(posixpath.basename(p), set()).add(p)
            if p.endswith(".py"):
                module = p[:-3].removesuffix("/__init__").replace("/", ".")
                self.by_module.setdefault(module, set()).add(p)
            parts = p.split("/")
            for depth in range(1, len(parts)):
                self.by_dir.setdefault("/".join(parts[:depth]), set()).add(p)

    def resolve(self, target: str) -> set[str]:
        target = posixpath.normpath(target)
        candidates = [target, *(target + e for e in _RESOLVE_EXTS)]
        candidates += [f"{target}/index{e}" for e in _RESOLVE_EXTS] + [f"{target}/__init__.py"]
        return {c for c in candidates if c in self.paths}

    def named_by(self, path: str, text: str, spent: set[tuple[str, str]]) -> set[str]:
        """Head paths ``text`` names. ``spent`` holds lookup keys already
        expanded for this closure: a key's hits are the same whichever file
        spells it, so each expands once (a 12k-file repo otherwise re-unions
        the same directory sets per file and takes minutes)."""
        hits: set[str] = set()
        prose = posixpath.splitext(path)[1].lower() in _PROSE_EXTS

        def take(table: str, key: str, found: set[str] | None) -> None:
            if found and (table, key) not in spent:
                spent.add((table, key))
                hits.update(found)

        for raw in set(_TOKEN.findall(text)):
            token = raw.strip("./").removeprefix("@/") if raw not in (".", "..") else ""
            if not token:
                continue
            base = posixpath.basename(token)
            # "basename WITH extension" is the contract: a bare word (``tests``,
            # ``version``) is prose, not a file reference, and treating it as
            # one would make every harness file ship-reachable through an
            # unrelated English mention (measured: 522/522 on this repo).
            if "." in base and not prose:
                take("basename", base, self.by_basename.get(base))
            take("module", token, self.by_module.get(token))
            if token in self.paths:
                hits.add(token)
            # A directory only counts when the file spells it as one
            # (``scripts/ci/``), never as a bare word or a path prefix.
            if token.endswith("/") and not prose:
                take("dir", token.rstrip("/"), self.by_dir.get(token.rstrip("/")))
        here = posixpath.dirname(path)
        for spec in _JS_SPEC.findall(text):
            hits |= self.resolve(posixpath.join(here, spec))
        for spec in _SH_SOURCE.findall(text):
            if spec.startswith("."):
                hits |= self.resolve(posixpath.join(here, spec))
        for dots, module, names in _PY_RELATIVE.findall(text):
            package = posixpath.normpath(posixpath.join(here, *([".."] * (len(dots) - 1))))
            base = posixpath.join(package, *module.split(".")) if module else package
            hits |= self.resolve(base)
            for name in re.findall(r"\w+", names):
                hits |= self.resolve(posixpath.join(base, name))
        hits.discard(path)
        return hits


def ship_closure(repo: Path, head: str, manifest: Manifest) -> dict[str, str]:
    """Map every closure file to how it entered: ``ship glob`` or the closure
    file that named it. Breadth-first, so the recorded referrer is shortest."""
    paths = [p for p in _git(repo, "ls-tree", "-r", "-z", "--name-only", head).split("\0") if p]
    index = _Index(paths)
    origin = {p: "ship glob" for p in paths if manifest.classify(p) == "ship"}
    frontier = list(origin)
    spent: set[tuple[str, str]] = set()
    while frontier:
        texts = _blobs(repo, head, frontier)
        frontier = []
        for path, text in texts.items():
            for named in sorted(index.named_by(path, text, spent)):
                if named not in origin:
                    origin[named] = path
                    frontier.append(named)
    return origin


# --- proof ----------------------------------------------------------------------


def _changed(repo: Path, a: str, b: str) -> list[str]:
    """Every path whose bytes or mode differ; renames as delete + add."""
    out = _git(repo, "diff", "--name-only", "-z", "--no-renames", "--no-ext-diff", a, b)
    return sorted({p for p in out.split("\0") if p})


def _split(repo: Path, mb: str, head: str, manifest: Manifest) -> tuple[int, int]:
    product = harness = 0
    out = _git(repo, "diff", "--numstat", "-z", "--no-renames", mb, head)
    for record in out.split("\0"):
        added, _, rest = record.partition("\t")
        if not rest:
            continue
        _, _, path = rest.partition("\t")
        lines = int(added) if added.isdigit() else 0  # binary: "-"
        if manifest.classify(path) == "harness":
            harness += lines
        else:
            product += lines
    return product, harness


def prove(
    repo: Path, reviewed: str, head: str, base: str, manifest_file: Path | None, split: bool
) -> tuple[bool, str]:
    reviewed_sha = _git(repo, "rev-parse", "--verify", f"{reviewed}^{{commit}}").strip()
    head_sha = _git(repo, "rev-parse", "--verify", f"{head}^{{commit}}").strip()
    base_sha = _git(repo, "rev-parse", "--verify", f"{base}^{{commit}}").strip()
    if manifest_file is not None:
        manifest = Manifest.parse(
            manifest_file.read_text(), f"external (replay): `{manifest_file}`"
        )
    else:
        try:
            text = _git(repo, "show", f"{head_sha}:{MANIFEST_PATH}")
        except ProofError:
            raise ProofError(
                f"no manifest: every delta is a round ({MANIFEST_PATH} absent at head)"
            ) from None
        manifest = Manifest.parse(text, f"in-repo `{MANIFEST_PATH}` @ head")

    mb_reviewed = _git(repo, "merge-base", base_sha, reviewed_sha).strip()
    mb_head = _git(repo, "merge-base", base_sha, head_sha).strip()
    changed = _changed(repo, reviewed_sha, head_sha)
    closure = ship_closure(repo, head_sha, manifest)

    reasons: list[str] = []
    if mb_reviewed != mb_head:
        reasons.append(
            f"base moved: merge-base `{mb_reviewed[:10]}` → `{mb_head[:10]}` "
            "(run scripts/rebase_proof.py; converge first)"
        )
    if manifest_file is None and MANIFEST_PATH in changed:
        reasons.append(f"manifest changed: `{MANIFEST_PATH}` is product; widening it needs a round")
    rows: list[str] = []
    for path in changed:
        kind = manifest.classify(path)
        if kind != "harness":
            if path != MANIFEST_PATH:
                reasons.append(f"product path changed: `{path}` ({kind})")
            rows.append(f"  - `{path}` — {kind}")
        elif path in closure:
            reasons.append(f"ship-reachable: `{path}` named by `{closure[path]}`")
            rows.append(f"  - `{path}` — harness, SHIP-REACHABLE via `{closure[path]}`")
        else:
            rows.append(f"  - `{path}` — harness")

    harness_only = not reasons
    verdict = (
        "HARNESS-ONLY — no numbered round; one `### Harness delta check` at the merge gate"
        if harness_only
        else "ROUND REQUIRED — " + "; ".join(reasons)
    )
    short = lambda sha: sha[:10]  # noqa: E731 — display only
    lines = [
        f"### Harness delta proof — {short(reviewed_sha)}..{short(head_sha)}",
        "",
        f"- Reviewed (clean round): `{short(reviewed_sha)}`; head: `{short(head_sha)}`",
        f"- Base: `{base}`; merge-bases `{short(mb_reviewed)}` → `{short(mb_head)}`"
        f" ({'same' if mb_reviewed == mb_head else 'MOVED'})",
        f"- Manifest: {manifest.source} "
        f"({len(manifest.harness)} harness, {len(manifest.ship)} ship globs)",
        f"- Ship closure at head: {len(closure)} files",
        f"- Changed paths ({len(changed)}):",
        *(rows or ["  - (none)"]),
    ]
    if split:
        product, harness = _split(repo, mb_head, head_sha, manifest)
        lines.append(
            f"- Added lines vs merge-base: product {product}, harness {harness} → "
            + ("SPLIT (harness to its own PR)" if harness > product else "NO SPLIT")
        )
    lines += [
        f"- Verdict: **{verdict}**",
        "",
        "Reproduce: `python scripts/harness_delta_proof.py "
        f"{short(reviewed_sha)} {short(head_sha)} --base {base}"
        + (f" --manifest {manifest_file}" if manifest_file else "")
        + (" --split" if split else "")
        + "`",
    ]
    return harness_only, "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("reviewed", help="head of the last clean (terminal) code round")
    parser.add_argument("head", help="current PR head")
    parser.add_argument("--base", default="origin/main", help="upstream ref (default origin/main)")
    parser.add_argument("--repo", default=".", help="repository path (default: cwd)")
    parser.add_argument("--manifest", type=Path, help="external manifest for retrospective replays")
    parser.add_argument(
        "--split", action="store_true", help="also print the measured split trigger"
    )
    args = parser.parse_args(argv)
    try:
        ok, block = prove(
            Path(args.repo), args.reviewed, args.head, args.base, args.manifest, args.split
        )
    except (ProofError, OSError) as exc:
        print(f"harness delta proof could not be computed: {exc}", file=sys.stderr)
        return 2
    print(block)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
