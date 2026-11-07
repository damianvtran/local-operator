#!/usr/bin/env python3
"""Prove a post-clean-round delta touches only test harness — or open a round.

Why this exists: local-operator-mobile PR #156 ran 40 numbered agent-review
rounds for a product change that was clean at round 3 (2026-11-03); the last
product-code commit landed 2026-11-05 02:14 and 20 failed native attempts plus
41.4 h of wall clock followed it. The rounds after round 3 reviewed evidence
harness (CI probes, fixtures, mock relays), not product. This script lets a
harness-only delta skip the numbered round (``guide://delivery``, "Harness-only
delta").

THE INVARIANT — the only thing this proves:

    No byte the shipping pipeline consumes differs between REVIEWED and HEAD.

HARNESS-ONLY (exit 0) requires ALL of:

1. Same base: ``merge-base(base, reviewed) == merge-base(base, head)``. A
   merged or rebased base puts upstream bytes into ``reviewed..head``; that is
   ``scripts/rebase_proof.py``'s job, never this one's.
2. The manifest (``.github/harness-paths``, then ``.gitlab/harness-paths``, then
   ``.harness-paths``) exists at head and is unchanged in ``reviewed..head``. The
   manifest is a product path: the exemption set cannot grow without a reviewed
   round. ``--manifest`` substitutes an external file for retrospective replays
   (labelled ``external (replay)``); then nothing in-repo is checked for
   condition 2.
3. Every changed path — added, modified, deleted, BOTH sides of a rename
   (``--no-renames``), and mode-only changes — classifies as ``harness``.
4. No changed path is ship-reachable. The ship closure is every file at head
   matching a ``ship:`` glob plus, transitively, every file a closure file
   NAMES: its repo path, its basename with extension, its dotted Python module,
   a directory spelled as one (``scripts/ci/``), a Python import, or a relative
   or aliased import/require/dynamic-import/``source`` specifier that resolves
   to it (tsconfig ``paths``/``baseUrl`` first, then .ts .tsx .js .mjs .cjs
   .json .py .sh and /index.*, then a trailing-segment match).

THE `gate:` TIER. A manifest may declare `gate: <exact path>` for a workflow or
the `Makefile` that runs checks but contributes no bytes to a shipped artifact. A
gate file is PRODUCT for change detection (editing it opens a round) and its text
seeds no ship closure, so the probes it runs are exempt-able; a changed harness
file that only a gate names is still shown as ``gate-run`` because the merge-gate
``### Harness delta check`` must read it with the full review lens, not only for
gate weakening. It fails closed: exit 2 when a gate's artifact, cache or job
output is consumed by a ship workflow, and a gate file a SHIP file names stops
being a gate (its text joins the closure).

WHAT THE CARVE-OUT IS NOT. It is a reachability classifier made safe by failing
closed — not a proof that a harness change cannot alter behaviour. It exempts
only manifest-declared harness files that nothing in the ship closure names, and
it says nothing about whether those files are correct or whether a workflow they
run in is sound; that is what the merge-gate ``### Harness delta check`` covers.

FAIL-CLOSED RULES (each one exists because an attack got past the first version;
each has a regression test):

- gitlink and ``.gitmodules`` changes are product — ``git diff`` respects
  ``.gitmodules ignore = all`` and the merger's own ``diff.ignoreSubmodules``,
  so every diff here passes ``--ignore-submodules=none`` and a mode-160000
  entry is never exempt.
- every symlink changed in the delta is product; a symlink inside the ship
  closure pulls its in-repo target — file or whole directory — into the closure,
  and one that leaves the repository or resolves to nothing is a round.
- blobs are read by object id from ``ls-tree -r -z`` (never by path, where a
  newline in a path misparses the batch); a short response, a missing object, an
  out-of-order reply or a header that is not a blob+sized blob is exit 2, never a
  skip.
- a ``harness:`` glob under ``.github/`` must be an exact filename: a workflow
  joins the shipping pipeline through its trigger, secrets and permissions, not
  by being named, so a wildcard there is rejected (exit 2) and an unlisted —
  hence new — workflow is product.
- a harness test file (``test_*.py``, ``*_test.py``, ``*.test.*``, ``*.spec.*``)
  that names or imports a ship-closure file is product: it guards shipped code.
- an unparsable tsconfig/alias file at head is exit 2 rather than a silent
  fallback to weaker matching.

Classification: a path matching both a ``harness:`` and a ``ship:`` glob takes
the MORE SPECIFIC glob (longer literal prefix before the first wildcard; ties go
to ship). That is what lets mobile keep ``harness: src/testing/**`` inside
``ship: src/*/**`` while local-operator keeps ``ship: .github/workflows/ci.yml``
beside its exact-filename harness workflows. A path matching neither is product.

Conservative by design: name matching is lexical, so a false positive opens a
round (the cost the gate already pays) while a false negative would ship an
unreviewed byte. Known lexical limit: a path assembled at runtime from fragments
no file spells out cannot be seen — declare such files under ``ship:``.

Exit 0 = HARNESS-ONLY, 1 = ROUND REQUIRED (each reason named), 2 = could not be
computed (bad ref, git failure, no manifest, unreadable object, rejected
manifest) — a missing proof is never a pass.

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
import json
import posixpath
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

#: Manifest search order: GitHub first, then GitLab, then the bare convention.
MANIFEST_PATHS = (".github/harness-paths", ".gitlab/harness-paths", ".harness-paths")
#: Extensions tried for an extensionless relative specifier, in resolver order.
_RESOLVE_EXTS = (".ts", ".tsx", ".js", ".mjs", ".cjs", ".json", ".py", ".sh")
_PROSE_EXTS = {".md", ".txt", ".rst"}
SYMLINK_MODE = "120000"
GITLINK_MODE = "160000"
#: Harness globs under this prefix must be exact filenames (see the docstring).
_GITHUB_PREFIX = ".github/"
#: Path-like words: the characters a repo path, basename or dotted module uses.
#: Everything else (quotes, ``${}``, ``:``, spaces, ``*``) separates tokens, so
#: ``${root}testing/cookie-jar-fetch.ts`` yields ``testing/cookie-jar-fetch.ts``.
_TOKEN = re.compile(r"[A-Za-z0-9_.@/+-]+")
#: JS/TS string specifiers, any quote style, ``./`` or ``../`` or an alias
#: (``@/…``). Template literals whose ``${...}`` interpolation only breaks the
#: literal apart still match: PR #156's fixture is reached as
#: ``${root}testing/cookie-jar-fetch.ts``.
_JS_SPEC = re.compile(r"""['"`](\.\.?/[^'"`\s]*|[@~]/[^'"`\s]*)['"`]""")
#: ``source x`` / ``. x`` shell include. LINEARITY IS A CONSTRAINT: no
#: quantified class here may straddle a newline — a failing match re-scans from
#: every line start, so a greedy multi-line class costs O(n^2) (measured: 36 min
#: of CPU on one 120 KB module before this shape).
_SH_SOURCE = re.compile(r"""(?m)(?:^|[;&|(])[ \t]*(?:source|\.)[ \t]+["']?([^\s;|&)'"]+)""")
#: Python relative import: ``from ..pkg.mod import a, b``. Same linearity rule:
#: the imported-name run stops at end of line, never crossing one.
_PY_RELATIVE = re.compile(r"""(?m)^[ \t]*from[ \t]+(\.+)([\w.]*)[ \t]+import[ \t]+([^\n]*)""")
#: Python absolute import / from-import: a sibling or package module by name.
_PY_IMPORT = re.compile(
    r"""(?m)^[ \t]*(?:from[ \t]+([\w.]+)[ \t]+import[ \t]+([^\n]*)|import[ \t]+([\w.]+))"""
)
#: Test filenames across the ecosystems this repo family uses.
_TEST_FILE = re.compile(
    r"(?:^|/)(?:test_[^/]+\.py|[^/]+_test\.(?:py|go)|[^/]+[.-]test\.[cm]?[jt]sx?|"
    r"[^/]+[.-]spec\.[cm]?[jt]sx?)$"
)


class ProofError(RuntimeError):
    """The proof could not be computed; the caller exits 2, never 0."""


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        errors="surrogateescape",
        check=False,
    )
    if result.returncode != 0:
        raise ProofError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def _strip_relative(token: str) -> str:
    """Drop leading ``./`` / ``../`` only.

    ``str.strip("./")`` also eats the dots of a dot-directory, so
    ``./.github/workflows/runner`` became ``github/workflows/runner`` and no
    reference under ``.github/`` ever matched (review R4 / QA Q2).
    """
    while token.startswith("./"):
        token = token[2:]
    while token.startswith("../"):
        token = token[3:]
    return token


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
    gate: list[str] = field(default_factory=list)
    source: str = ""

    @classmethod
    def parse(cls, text: str, source: str) -> Manifest:
        manifest = cls(source=source)
        for number, raw in enumerate(text.splitlines(), 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            kind, sep, glob = line.partition(":")
            if not sep or kind.strip() not in ("harness", "ship", "gate") or not glob.strip():
                raise ProofError(
                    f"manifest line {number} is not 'harness: <glob>', 'ship: <glob>' "
                    "or 'gate: <exact path>'"
                )
            kind, glob = kind.strip(), glob.strip()
            if kind == "gate" and re.search(r"[*?]", glob):
                raise ProofError(
                    f"manifest line {number}: a gate entry must be an exact path (got "
                    f"'{glob}'): the check that it feeds no shipped artifact is per file"
                )
            if kind == "harness" and glob.startswith(_GITHUB_PREFIX) and re.search(r"[*?]", glob):
                raise ProofError(
                    f"manifest line {number}: a harness glob under {_GITHUB_PREFIX} must be an "
                    f"exact filename (got '{glob}'): a workflow joins the shipping pipeline "
                    "through its trigger, secrets and permissions, not by being named"
                )
            getattr(manifest, kind).append(glob)
        return manifest

    def classify(self, path: str) -> str:
        """``ship`` | ``harness`` | ``gate`` | ``product``.

        A ``gate:`` entry is checked first and wins over every glob: a gate file
        runs checks but ships no bytes, and it is PRODUCT for change detection —
        editing it always opens a round (round-2 design, ``gate:`` tier).
        """
        if path in self.gate:
            return "gate"

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


# --- object reading (by id, never by path) --------------------------------------


def _tree(repo: Path, rev: str) -> dict[str, tuple[str, str]]:
    """``path -> (mode, object id)`` for every non-tree entry at ``rev``."""
    out = _git(repo, "ls-tree", "-r", "-z", rev)
    tree: dict[str, tuple[str, str]] = {}
    for record in out.split("\0"):
        if not record:
            continue
        meta, _, path = record.partition("\t")
        mode, _kind, oid = meta.split()
        tree[path] = (mode, oid)
    return tree


def _changed(repo: Path, a: str, b: str) -> dict[str, tuple[str, str, str, str]]:
    """``path -> (old_mode, new_mode, old_oid, new_oid)``; renames as delete+add.

    ``--ignore-submodules=none`` is load-bearing: without it a gitlink bump is
    absent from the diff entirely when ``.gitmodules`` sets ``ignore = all`` or
    the merger's git config sets ``diff.ignoreSubmodules`` (review R1), which
    would let a product-path entry change under a HARNESS-ONLY verdict.
    """
    raw = _git(
        repo,
        "diff",
        "--raw",
        "--no-abbrev",
        "-z",
        "--no-renames",
        "--no-ext-diff",
        "--ignore-submodules=none",
        a,
        b,
    )
    fields = raw.split("\0")
    changes: dict[str, tuple[str, str, str, str]] = {}
    index = 0
    while index + 1 < len(fields):
        meta, path = fields[index], fields[index + 1]
        index += 2
        if not meta.startswith(":"):
            raise ProofError(f"unexpected `git diff --raw` record: {meta!r}")
        try:
            old_mode, new_mode, old_oid, new_oid, _status = meta[1:].split()
        except ValueError as exc:  # pragma: no cover - git format change
            raise ProofError(f"unparsable `git diff --raw` record: {meta!r}") from exc
        changes[path] = (old_mode, new_mode, old_oid, new_oid)
    return changes


def _blobs(repo: Path, oids: set[str]) -> dict[str, str]:
    """Blob text by object id, in one ``cat-file --batch``.

    Reading by id rather than ``rev:path`` is what makes a path containing a
    newline safe: the batch protocol speaks ids and sizes, so a path can never
    desynchronise it. Every mismatch raises — a proof that cannot read the bytes
    it is classifying must not report HARNESS-ONLY (review R2).
    """
    if not oids:
        return {}
    ordered = sorted(oids)
    request = "".join(f"{oid}\n" for oid in ordered).encode()
    proc = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "--batch"],
        input=request,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise ProofError(f"git cat-file --batch failed: {proc.stderr.decode().strip()}")
    data, pos, out = proc.stdout, 0, {}
    for oid in ordered:
        newline = data.find(b"\n", pos)
        if newline < 0:
            raise ProofError("git cat-file --batch: truncated response")
        header = data[pos:newline].split()
        pos = newline + 1
        if len(header) == 2 and header[1] == b"missing":
            raise ProofError(f"git cat-file: object {oid} is missing")
        if len(header) != 3:
            raise ProofError(f"git cat-file: unparsable header for {oid}")
        got, kind, size = header[0].decode(), header[1], int(header[2])
        if got != oid:
            raise ProofError(f"git cat-file: out-of-order response ({got} for {oid})")
        body = data[pos : pos + size]
        if len(body) != size:
            raise ProofError(f"git cat-file: short read for {oid}")
        pos += size + 1  # trailing LF
        if kind == b"blob":
            out[oid] = body.decode("latin-1")
        elif kind != b"commit":  # gitlink target: nothing to scan, but loud
            raise ProofError(f"git cat-file: {oid} is a {kind.decode()}, not a blob")
    return out


# --- ship closure ---------------------------------------------------------------


def _tsconfig_aliases(
    repo: Path, head: str, tree: dict[str, tuple[str, str]]
) -> tuple[str, dict[str, list[str]], str | None]:
    """``(baseUrl, paths, problem)`` from the head's tsconfig/jsconfig.

    Aliased imports (``@/testing/helper``) name no repo path, so the token scan
    cannot see them; ``compilerOptions.paths``/``baseUrl`` is the authoritative
    mapping when the repo carries one (review M1 / QA Q3). A file that cannot be
    parsed is a problem, not a silent downgrade to weaker matching.
    """
    for name in ("tsconfig.json", "jsconfig.json"):
        if name not in tree:
            continue
        text = _blobs(repo, {tree[name][1]}).get(tree[name][1], "")
        try:
            data = json.loads(re.sub(r"(?m)^\s*//.*$", "", text))
        except json.JSONDecodeError as exc:
            return (
                ".",
                {},
                f"`{name}` at head cannot be parsed ({exc.msg}); alias imports unreadable",
            )
        compiler = data.get("compilerOptions") or {}
        paths = {
            str(k): [str(v) for v in (vals if isinstance(vals, list) else [vals])]
            for k, vals in (compiler.get("paths") or {}).items()
        }
        return str(compiler.get("baseUrl") or "."), paths, None
    return ".", {}, None


class _Index:
    """Lookup tables from a name a file can carry to the head paths it denotes."""

    def __init__(
        self,
        paths: list[str],
        base_url: str = ".",
        aliases: dict[str, list[str]] | None = None,
    ) -> None:
        self.paths = set(paths)
        self.base_url = base_url
        self.aliases = aliases or {}
        self.by_basename: dict[str, set[str]] = {}
        self.by_module: dict[str, set[str]] = {}
        self.by_dir: dict[str, set[str]] = {}
        self.by_suffix: dict[str, set[str]] = {}
        for p in paths:
            self.by_basename.setdefault(posixpath.basename(p), set()).add(p)
            if p.endswith(".py"):
                module = p[:-3].removesuffix("/__init__").replace("/", ".")
                self.by_module.setdefault(module, set()).add(p)
            parts = p.split("/")
            for depth in range(1, len(parts)):
                self.by_dir.setdefault("/".join(parts[:depth]), set()).add(p)
            # Trailing-segment keys: an aliased specifier names no repo path but
            # its tail often equals one (M1's conservative fallback).
            stem = posixpath.splitext(p)[0].split("/")
            for depth in range(2, min(5, len(stem) + 1)):
                self.by_suffix.setdefault("/".join(stem[-depth:]), set()).add(p)

    def _resolve_path(self, target: str) -> set[str]:
        target = posixpath.normpath(target)
        if target.startswith(".."):
            return set()
        candidates = [target, *(target + e for e in _RESOLVE_EXTS)]
        candidates += [f"{target}/index{e}" for e in _RESOLVE_EXTS] + [f"{target}/__init__.py"]
        return {c for c in candidates if c in self.paths}

    def _alias_targets(self, spec: str) -> list[str]:
        out = []
        for pattern, targets in self.aliases.items():
            if "*" in pattern:
                prefix, _, suffix = pattern.partition("*")
                if (
                    spec.startswith(prefix)
                    and spec.endswith(suffix)
                    and len(spec) >= len(prefix + suffix)
                ):
                    middle = spec[len(prefix) : len(spec) - len(suffix)]
                    out.extend(t.replace("*", middle) for t in targets)
            elif spec == pattern:
                out.extend(targets)
        return [posixpath.join(self.base_url, t) for t in out]

    def resolve_module(self, name: str) -> set[str]:
        """A Python import's file: exact path first, then any file of that name.

        ``import helper`` from a package module means "wherever sys.path finds
        it", so the conservative reading is any head ``helper.py`` (or
        ``helper/__init__.py``). Missing a reachable harness file here would let
        a shipped module import unreviewed harness code (review M2).
        """
        rel = name.replace(".", "/")
        hits = self._resolve_path(rel)
        last = posixpath.basename(rel)
        hits |= self.by_basename.get(last + ".py", set())
        hits |= self.by_basename.get(last + "/__init__.py", set())
        if not hits:
            hits |= self.by_suffix.get(rel.strip("/"), set())
        return hits

    def resolve_specifier(self, spec: str) -> set[str]:
        """Head paths a JS/TS-style specifier denotes."""
        spec = _strip_relative(spec.split("?")[0].split("#")[0])
        if not spec:
            return set()
        hits: set[str] = set()
        for target in self._alias_targets(spec):
            hits |= self._resolve_path(target)
        bare = spec[2:] if spec[:2] in ("@/", "~/") else spec
        for candidate in (spec, bare):
            hits |= self._resolve_path(candidate)
        for candidate in (spec, bare):
            stem = posixpath.splitext(candidate)[0]
            parts = [p for p in stem.split("/") if p and p != "."]
            for depth in range(min(4, len(parts)), 1, -1):
                found = self.by_suffix.get("/".join(parts[-depth:]))
                if found:
                    hits |= found
                    break
        return hits

    def named_by(self, path: str, text: str, spent: set[tuple[str, str]]) -> set[str]:
        """Head paths ``text`` names. ``spent`` holds lookup keys already
        expanded for this closure: a key's hits are the same whichever file
        spells it, so each expands once (a 12k-file repo otherwise re-unions the
        same directory sets per file and takes minutes)."""
        hits: set[str] = set()
        prose = posixpath.splitext(path)[1].lower() in _PROSE_EXTS

        def take(table: str, key: str, found: set[str] | None) -> None:
            if found and (table, key) not in spent:
                spent.add((table, key))
                hits.update(found)

        for raw in set(_TOKEN.findall(text)):
            if raw in (".", ".."):
                continue
            spelled_dir = raw.endswith("/")
            token = _strip_relative(raw).rstrip("/")
            if not token:
                continue
            base = posixpath.basename(token)
            # "basename WITH extension" is the contract: a bare word (``tests``,
            # ``version``) is prose, not a file reference, and treating it as one
            # made every harness file reachable through an unrelated mention
            # (measured: 522/522 on this repo).
            if "." in base and not prose:
                take("basename", base, self.by_basename.get(base))
            take("module", token, self.by_module.get(token))
            if token in self.paths:
                hits.add(token)
            # A directory counts only when the file SPELLS it as one
            # (``scripts/ci/``, trailing slash). A bare directory word does not:
            # publish.yml's prose ``scripts`` pulled all 327 files under
            # scripts/ into the ship closure and made every probe ship-reachable
            # (measured round 2), and a word is not a path reference.
            if spelled_dir:
                take("dir", token, self.by_dir.get(token))
        here = posixpath.dirname(path)
        for spec in _JS_SPEC.findall(text):
            if spec.startswith((".", "@", "~")):
                if spec[0] == ".":
                    rel = posixpath.join(here, spec)
                    hits |= self._resolve_path(rel)
                    hits |= self.resolve_specifier(spec)
                else:
                    hits |= self.resolve_specifier(spec)
        for spec in _SH_SOURCE.findall(text):
            if spec.startswith("."):
                hits |= self._resolve_path(posixpath.join(here, spec))
        for dots, module, names in _PY_RELATIVE.findall(text):
            package = posixpath.normpath(posixpath.join(here, *([".."] * (len(dots) - 1))))
            base = posixpath.join(package, *module.split(".")) if module else package
            hits |= self._resolve_path(base)
            for name in re.findall(r"\w+", names):
                hits |= self._resolve_path(posixpath.join(base, name))
        for absolute, names, single in _PY_IMPORT.findall(text):
            module = (absolute or single).replace(".", "/")
            hits |= self.resolve_module(absolute or single)
            if absolute:
                for name in re.findall(r"\w+", names):
                    hits |= self._resolve_path(posixpath.join(module, name))
        hits.discard(path)
        return hits


@dataclass
class Closure:
    """The ship closure and everything that went wrong building it."""

    origin: dict[str, str]
    problems: list[str]
    index: _Index
    gate_texts: dict[str, str] = field(default_factory=dict)


def ship_closure(repo: Path, head: str, manifest: Manifest) -> Closure:
    """Every file the shipping pipeline can read at ``head``, and why it joined.

    Breadth-first, so the recorded referrer is the shortest chain. Symlinks are
    expanded rather than scanned as text: a ship symlink to a harness *directory*
    made that directory's whole contents invisible to the first version while
    setuptools packaged them (review R3).
    """
    tree = _tree(repo, head)
    base_url, aliases, problem = _tsconfig_aliases(repo, head, tree)
    gate_paths = set(manifest.gate)
    index = _Index(list(tree), base_url, aliases)
    origin: dict[str, str] = {p: "ship glob" for p in tree if manifest.classify(p) == "ship"}
    problems: list[str] = [problem] if problem else []
    scanned: set[str] = set()
    links_done: set[str] = set()
    spent: set[tuple[str, str]] = set()
    while True:
        fresh = [p for p in origin if p not in scanned]
        scanned.update(fresh)
        scannable = [p for p in fresh if tree[p][0] not in (SYMLINK_MODE, GITLINK_MODE)]
        blobs = _blobs(repo, {tree[p][1] for p in scannable})
        for path in fresh:
            if tree[path][0] == SYMLINK_MODE:
                continue
            text = blobs.get(tree[path][1])
            if text is None:
                continue
            for named in sorted(index.named_by(path, text, spent)):
                # A gate file a SHIP file names (a reusable workflow called via
                # `uses:`) stops being a pure gate: its text joins the closure and
                # everything it names becomes ship-reachable — the exemption
                # simply evaporates, no error. A doc that merely mentions the
                # gate ("see the Makefile") does not escalate it: docs name the
                # tooling constantly and that mention ships no bytes.
                if named in gate_paths and manifest.classify(path) != "ship":
                    continue
                if named not in origin:
                    origin[named] = path
        added_link = False
        for path in [p for p in origin if tree[p][0] == SYMLINK_MODE and p not in links_done]:
            links_done.add(path)
            target = _blobs(repo, {tree[path][1]}).get(tree[path][1], "").strip()
            resolved = (
                posixpath.normpath(posixpath.join(posixpath.dirname(path), target))
                if target
                else ""
            )
            if not target or target.startswith("/") or resolved.startswith(".."):
                problems.append(
                    f"ship symlink `{path}` -> `{target or '(empty)'}` leaves the repository"
                )
                continue
            if resolved in tree:
                under = [resolved]
            else:
                under = sorted(p for p in tree if p.startswith(resolved + "/"))
            if not under:
                problems.append(f"ship symlink `{path}` -> `{target}` resolves to nothing at head")
                continue
            for entry in under:
                if entry not in origin:
                    origin[entry] = f"symlink `{path}`"
                    added_link = True
        if not fresh and not added_link:
            break
    gate_texts = {
        gate: _blobs(repo, {tree[gate][1]}).get(tree[gate][1], "")
        for gate in manifest.gate
        if gate in tree and tree[gate][0] not in (SYMLINK_MODE, GITLINK_MODE)
    }
    _gate_feeds_ship(repo, tree, origin, manifest, gate_texts)
    return Closure(origin, problems, index, gate_texts)


#: An upload/download step's artifact name, and a cache step's key, within the
#: handful of lines a workflow step body occupies.
_ARTIFACT_NAME = re.compile(r"(?m)^\s*name:\s*([A-Za-z0-9_.${}/-]+)\s*$")
_CACHE_KEY = re.compile(r"(?m)^\s*key:\s*([A-Za-z0-9_.${}/-]+)\s*$")


def _step_values(text: str, action: str) -> set[str]:
    """``name:`` values belonging to steps that use ``action`` (cache: ``key:``)."""
    field = _CACHE_KEY if "cache" in action else _ARTIFACT_NAME
    values: set[str] = set()
    lines = text.splitlines()
    for number, line in enumerate(lines):
        if action not in line:
            continue
        for follow in lines[number + 1 : number + 9]:
            found = field.search(follow)
            if found:
                values.add(found.group(1))
                break
            if "uses:" in follow:
                break
    return values


def _gate_feeds_ship(
    repo: Path,
    tree: dict[str, tuple[str, str]],
    origin: dict[str, str],
    manifest: Manifest,
    gate_texts: dict[str, str],
) -> list[str]:
    """A gate may not feed an artifact, cache or job output INTO a ship workflow.

    The point of the tier is that a gate decides whether shipping proceeds while
    contributing no bytes; the moment a ship workflow downloads what a gate
    uploaded (or restores its cache, or reads its job outputs), the gate's text is
    shipping input and the tier would be a hole. The comparison is by NAME, not by
    the mere presence of ``upload-artifact``: publish.yml downloads artifacts its
    own jobs uploaded, which says nothing about a gate.
    """
    if not gate_texts:
        return
    # Consumers are files the shipping pipeline actually runs — those the
    # manifest classifies `ship:` — not every file the closure reaches: a test
    # that asserts a workflow's wiring quotes `needs.x.outputs` without consuming
    # it, and reading that as a feed would make every gate un-declarable.
    shipped = {
        path: oid
        for path, (mode, oid) in tree.items()
        if path in origin
        and path not in gate_texts
        and mode not in (SYMLINK_MODE, GITLINK_MODE)
        and manifest.classify(path) == "ship"
    }
    texts = _blobs(repo, set(shipped.values()))
    for gate, gate_text in gate_texts.items():
        uploads = _step_values(gate_text, "upload-artifact")
        caches = _step_values(gate_text, "actions/cache")

        for path, oid in shipped.items():
            text = texts.get(oid)
            if text is None:
                continue
            if uploads and "download-artifact" in text:
                shared = uploads & _step_values(text, "download-artifact")
                if shared:
                    raise ProofError(
                        f"gate file feeds ship: `{gate}` uploads {sorted(shared)[0]!r}, "
                        f"which `{path}` downloads"
                    )
            if caches and "actions/cache" in text:
                shared = caches & _step_values(text, "actions/cache")
                if shared:
                    raise ProofError(
                        f"gate file feeds ship: `{gate}` caches {sorted(shared)[0]!r}, "
                        f"which `{path}` restores"
                    )
            # `needs.<job>.outputs` only ever refers to a job in the SAME
            # workflow, so it cannot be a cross-file feed — a ship workflow with
            # its own job of that name tripped this rule (mobile android.yml).
            # Cross-workflow consumption is `workflow_run` on the gate, which
            # makes the gate's run shipping input.
            # `workflow_run.workflows` names the workflow's `name:` or its file,
            # so both spellings count.
            gate_names = {
                posixpath.basename(gate),
                posixpath.basename(gate).rsplit(".", 1)[0],
            }
            if "workflow_run:" in text and any(
                re.search(rf"workflows:[^\n]*{re.escape(name)}", text) for name in gate_names
            ):
                raise ProofError(
                    f"gate file feeds ship: `{path}` triggers on `workflow_run` of gate "
                    f"`{gate}`, so the gate's run is shipping input"
                )


# --- proof ----------------------------------------------------------------------


def _split(repo: Path, mb: str, head: str, manifest: Manifest) -> tuple[int, int]:
    product = harness = 0
    out = _git(
        repo, "diff", "--numstat", "-z", "--no-renames", "--ignore-submodules=none", mb, head
    )
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


def _gate_names(path: str, closure: Closure) -> str | None:
    """The gate file that names ``path``, if any (a label, never a pass)."""
    for gate, text in closure.gate_texts.items():
        if path in closure.index.named_by(gate, text, set()):
            return gate
    return None


def _test_guards_ship(
    repo: Path,
    changes: dict[str, tuple[str, str, str, str]],
    closure: Closure,
    path: str,
) -> str | None:
    """Reviewer M1 / QA Q-M1: a test of shipped code is product, not harness."""
    _old_mode, _new_mode, old_oid, new_oid = changes[path]
    # A deletion carries the all-zero id on the head side, not an empty string:
    # reading it made a deleted harness test exit 2 instead of returning a
    # verdict (found replaying PR #156 rounds 27 and 36). Read the reviewed
    # side — its guard on shipped code is exactly what deleting it threatens.
    oid = new_oid if new_oid.strip("0") else old_oid
    text = _blobs(repo, {oid}).get(oid)
    if text is None:
        return f"harness test `{path}` cannot be read; its ship guard is unknown"
    named = closure.index.named_by(path, text, set()) & set(closure.origin)
    if named:
        referent = sorted(named)[0]
        return f"harness test `{path}` guards ship file `{referent}`"
    return None


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
        manifest_label = manifest.source
    else:
        text = label = None
        for candidate in MANIFEST_PATHS:
            try:
                text = _git(repo, "show", f"{head_sha}:{candidate}")
            except ProofError:
                continue
            label = candidate
            break
        if text is None or label is None:
            raise ProofError(
                "no manifest: every delta is a round "
                f"(none of {', '.join(MANIFEST_PATHS)} at head)"
            )
        manifest = Manifest.parse(text, f"in-repo `{label}` @ head")
        manifest_label = label

    mb_reviewed = _git(repo, "merge-base", base_sha, reviewed_sha).strip()
    mb_head = _git(repo, "merge-base", base_sha, head_sha).strip()

    changes = _changed(repo, reviewed_sha, head_sha)
    closure = ship_closure(repo, head_sha, manifest)

    reasons: list[str] = []
    if mb_reviewed != mb_head:
        reasons.append(
            f"base moved: merge-base `{mb_reviewed[:10]}` -> `{mb_head[:10]}` "
            "(run scripts/rebase_proof.py; converge first)"
        )
    if manifest_file is None and manifest_label in changes:
        reasons.append(
            f"manifest changed: `{manifest_label}` is product; widening it needs a round"
        )
    reasons.extend(f"fail-closed: {problem}" for problem in closure.problems)

    rows: list[str] = []
    gate_runs: list[str] = []
    for path in sorted(changes):
        old_mode, new_mode, _old_oid, _new_oid = changes[path]
        kind = manifest.classify(path)
        if GITLINK_MODE in (old_mode, new_mode):
            reasons.append(f"product path changed: `{path}` (gitlink {old_mode}->{new_mode})")
            rows.append(f"  - `{path}` — gitlink, product")
            continue
        if SYMLINK_MODE in (old_mode, new_mode):
            reasons.append(f"product path changed: `{path}` (symlink {old_mode}->{new_mode})")
            rows.append(f"  - `{path}` — symlink, product")
            continue
        if path == ".gitmodules":
            reasons.append(f"product path changed: `{path}` (submodule config)")
            rows.append(f"  - `{path}` — submodule config, product")
            continue
        if kind != "harness":
            if path != manifest_label:
                reasons.append(f"product path changed: `{path}` ({kind})")
            rows.append(f"  - `{path}` — {kind}")
            continue
        if path in closure.origin:
            reasons.append(f"ship-reachable: `{path}` named by `{closure.origin[path]}`")
            rows.append(f"  - `{path}` — harness, SHIP-REACHABLE via `{closure.origin[path]}`")
            continue
        if _TEST_FILE.search(path):
            guard = _test_guards_ship(repo, changes, closure, path)
            if guard:
                reasons.append(f"product path changed: `{guard}`")
                rows.append(f"  - `{path}` — harness test, product: {guard}")
                continue
        gate_by = _gate_names(path, closure)
        if gate_by:
            gate_runs.append(f"`{path}` (via `{gate_by}`)")
            rows.append(f"  - `{path}` — harness, gate-run via `{gate_by}`")
            continue
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
        f"- Base: `{base}`; merge-bases `{short(mb_reviewed)}` -> `{short(mb_head)}`"
        f" ({'same' if mb_reviewed == mb_head else 'MOVED'})",
        f"- Manifest: {manifest.source} "
        f"({len(manifest.harness)} harness, {len(manifest.ship)} ship globs)",
        f"- Ship closure at head: {len(closure.origin)} files",
        "- Invariant: no byte the shipping pipeline consumes differs between "
        f"`{short(reviewed_sha)}` and `{short(head_sha)}`",
        "- Fail-closed rules applied: gitlink/.gitmodules → product; symlink change → product; "
        "ship symlink leaving the repo → round; unreadable object → exit 2; "
        "`.github/` harness globs must be exact filenames; harness test guarding ship code → "
        "product; added workflow unmatched by the manifest → product",
        f"- Gate tier: {len(manifest.gate)} gate file(s) seed no ship closure; "
        + (
            "gate-run harness files — "
            + ", ".join(sorted(gate_runs))
            + " — must be read at the merge gate with the FULL review lens, not only for "
            "gate weakening"
            if gate_runs
            else "no changed harness file is gate-run"
        ),
        f"- Changed paths ({len(changes)}):",
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
