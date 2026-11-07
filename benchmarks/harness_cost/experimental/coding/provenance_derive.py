"""Derive public image inputs from a task's OWN build sources (host-side, stdlib).

WHY (oracle sweep a18da995, S9): the gold-isolation scan refuses any readable
file whose whole-file hash equals a task ``solution/`` or ``tests/`` file. Some
Terminal-Bench images ship such bytes on purpose -- the Dockerfile COPYs a
tests fixture into ``/app`` for the agent to check its work (schemelike,
circuit-fibsqrt, overfull-hbox, tune-mjcf), or a build script clones a public
repository whose files the tests reuse (sanitize-git-repo). Those bytes are
public by construction, so every arm sees them; refusing them refuses the task
for all seven arms before inference.

This module turns the task's explicit build inputs into exact
``destination -> sha256`` pairs with a file:line justification each, which are
then FROZEN in ``public-provenance.json`` and reviewed. It never inspects a
live container: an observed dirty image must not grant itself permissions.

CONSTRAINTS (fail closed rather than guess):

* Dockerfile: only the FINAL stage counts; ``COPY``/``ADD`` from the build
  context (``environment/``) with literal sources. ``--from=`` copies, globs,
  URLs, ``ADD`` archives, ``.dockerignore`` and relative destinations without
  a known ``WORKDIR`` raise ``Unsupported`` instead of producing pairs.
* Build-time clones: only the narrow, reviewable shape
  ``git clone <https-url> <dir>`` followed by ``cd <dir>`` and a hard pin
  ``[ "$(git rev-parse HEAD)" = "<40-hex>" ] || exit 1`` in a script the final
  stage both COPYs and RUNs. A clone without that pin is reported as a
  disclosed limitation and contributes NO pairs (a moving ref is not
  reviewable provenance). The pinned tree's bytes come from a host-side clone
  at exactly that commit, verified by ``git rev-parse``; git's content
  addressing binds the bytes to the pinned commit.
* Only pairs whose digest is in the guarded gold set are emitted: a public
  file that matches nothing hidden needs no qualification, and a narrower
  inventory is easier to review. Uncopied fixtures (e.g. schemelike's
  ``shadow_test``) are never emitted, so they stay on the hidden deny-list.

Usage (freeze, then review the diff of public-provenance.json)::

    python3 provenance_derive.py <task_dir> <image_digest> [--clone-cache DIR]
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path, PurePosixPath

PIN = re.compile(r'^\[\s+"\$\(git rev-parse HEAD\)"\s+=\s+"([0-9a-f]{40})"\s+\]\s+\|\|\s+exit\s+1$')
CLONE = re.compile(r"^git clone (https://[A-Za-z0-9./_-]+?(?:\.git)?) ([A-Za-z0-9._-]+)$")


class Unsupported(ValueError):
    """A build input this deriver will not interpret; no pairs are produced."""


def sha256(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def _instructions(dockerfile: Path):
    """``(line_no, KEYWORD, rest)`` for the final stage, continuations joined."""
    stage, pending, start = [], "", 0
    for no, raw in enumerate(dockerfile.read_text().splitlines(), 1):
        line = raw.strip()
        if not pending and (not line or line.startswith("#")):
            continue
        if not pending:
            start = no
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        text, pending = (pending + line).strip(), ""
        keyword, _, rest = text.partition(" ")
        if keyword.upper() == "FROM":
            stage = []
        stage.append((start, keyword.upper(), rest.strip()))
    return stage


def _copy_args(rest: str) -> tuple[list[str], list[str]]:
    words = json.loads(rest) if rest.startswith("[") else shlex.split(rest)
    flags = [w for w in words if w.startswith("--")]
    return flags, [w for w in words if not w.startswith("--")]


def _copy_pairs(task_dir: Path, gold: set[str]) -> tuple[list[dict], list[tuple[int, str]]]:
    env = task_dir / "environment"
    if (env / ".dockerignore").exists():
        raise Unsupported(".dockerignore changes the build context")
    workdir, dirs, pairs, runs = None, set(), [], []
    for no, keyword, rest in _instructions(env / "Dockerfile"):
        cite = f"environment/Dockerfile:{no}: {keyword} {rest}"
        if keyword == "WORKDIR":
            path = PurePosixPath(rest) if rest.startswith("/") else (
                PurePosixPath(workdir or "/") / rest)
            workdir = str(path)
            dirs.add(workdir)
        elif keyword == "RUN":
            # The script's cwd is the WORKDIR in force at THIS RUN line.
            runs.append((no, rest, workdir))
        elif keyword in ("COPY", "ADD"):
            flags, words = _copy_args(rest)
            if any(f.startswith("--from") for f in flags):
                raise Unsupported(f"{cite}: copy from another stage/image")
            *sources, dest = words
            if any(re.search(r"[*?\[]|://", s) for s in sources):
                raise Unsupported(f"{cite}: glob or remote source")
            if not dest.startswith("/"):
                if workdir is None:
                    raise Unsupported(f"{cite}: relative destination without WORKDIR")
                dest = str(PurePosixPath(workdir) / dest)
            dest = str(PurePosixPath(dest))
            for source in sources:
                src = (env / source).resolve()
                if not src.is_relative_to(env.resolve()) or not src.exists():
                    raise Unsupported(f"{cite}: source outside the build context")
                if keyword == "ADD" and src.is_file() and re.search(
                        r"\.(tar|tgz|gz|bz2|xz|zst)$", src.name):
                    raise Unsupported(f"{cite}: ADD auto-extracts archives")
                if src.is_dir():
                    files = [(p, str(PurePosixPath(dest) / p.relative_to(src).as_posix()))
                             for p in sorted(src.rglob("*")) if p.is_file()]
                    dirs.add(dest)
                else:
                    into = (len(sources) > 1 or words[-1].endswith(("/", "."))
                            or dest in dirs)
                    files = [(src, str(PurePosixPath(dest) / src.name) if into else dest)]
                for path, target in files:
                    digest = sha256(path)
                    if digest in gold:
                        pairs.append({"origin": "dockerfile-copy",
                                      "source": str(path.relative_to(task_dir)),
                                      "path": target, "sha256": digest,
                                      "justification": cite})
    return pairs, runs


def _clones(task_dir: Path, runs) -> list[dict]:
    """Recognize pinned ``git clone`` in a script the final stage RUNs."""
    found = []
    for script in sorted((task_dir / "environment").glob("*.sh")):
        ran = [(no, cwd) for no, cmd, cwd in runs
               if f"/{script.name}" in cmd or cmd.endswith(script.name)]
        if not ran:
            continue
        lines = script.read_text().splitlines()
        for no, line in enumerate(lines, 1):
            match = CLONE.match(line.strip())
            if not match:
                if line.strip().startswith("git clone"):
                    raise Unsupported(f"environment/{script.name}:{no}: unrecognized clone")
                continue
            url, name = match.groups()
            pin = None
            for later_no, later in enumerate(lines[no:], no + 1):
                if later.strip() == f"cd {name}":
                    nxt = lines[later_no] if later_no < len(lines) else ""
                    pinned = PIN.match(nxt.strip())
                    if pinned:
                        pin = (pinned.group(1), later_no + 1)
                    break
            found.append({"script": f"environment/{script.name}", "line": no, "url": url,
                          "dir": name, "run_line": ran[0][0], "cwd": ran[0][1], "pin": pin})
    return found


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, timeout=300).stdout


def clone_pairs(clone: dict, cache: Path, gold: set[str]) -> list[dict]:
    """Pairs from a host-side checkout of the pinned commit (fetched if absent)."""
    commit, pin_line = clone["pin"]
    if clone["cwd"] is None:
        raise Unsupported(f"{clone['script']}: clone run without a known WORKDIR")
    workdir = clone["cwd"]
    repo = cache / f"{clone['dir']}-{commit}"
    if not repo.exists():
        subprocess.run(["git", "clone", "--quiet", "--no-checkout", clone["url"], str(repo)],
                       check=True, timeout=600)
    if _git(repo, "rev-parse", f"{commit}^{{commit}}").decode().strip() != commit:
        raise Unsupported("pinned commit not resolvable in the clone cache")
    origin = f"build-time clone of {clone['url']}@{commit}"
    cite = (f"{clone['script']}:{clone['line']} (git clone), {clone['script']}:{pin_line} "
            f"(HEAD pin), environment/Dockerfile:{clone['run_line']} (RUN)")
    pairs = []
    for row in _git(repo, "ls-tree", "-r", "-z", commit).split(b"\0"):
        if not row:
            continue
        meta, path = row.decode().split("\t", 1)
        mode, kind, blob = meta.split()
        if kind != "blob" or mode not in ("100644", "100755"):
            continue
        digest = hashlib.sha256(_git(repo, "cat-file", "blob", blob)).hexdigest()
        if digest in gold:
            pairs.append({"origin": origin, "build_input": clone["script"],
                          "path": str(PurePosixPath(workdir) / clone["dir"] / path),
                          "sha256": digest, "git_blob": blob, "justification": cite})
    return pairs


def derive(task_dir: Path, gold: set[str], clone_cache: Path | None = None) -> dict:
    """Public pairs plus disclosed limitations, for review then freezing."""
    pairs, runs = _copy_pairs(task_dir, gold)
    limitations = []
    for clone in _clones(task_dir, runs):
        if clone["pin"] is None:
            limitations.append(
                f"{clone['script']}:{clone['line']}: clone of {clone['url']} is not pinned "
                "to a commit; its files get no public provenance (stay hidden-denied)")
        elif clone_cache is None:
            raise Unsupported("a pinned clone needs --clone-cache to derive its pairs")
        else:
            pairs += clone_pairs(clone, clone_cache, gold)
    paths = [p["path"] for p in pairs]
    if len(paths) != len(set(paths)):
        raise Unsupported("two build inputs claim the same destination")
    return {"public_files": sorted(pairs, key=lambda p: p["path"]),
            "limitations": limitations}


# S9c post-agent rule (R11 F27): NOTHING is auto-classified as disclosable.
# A hidden tests/ file may be disclosed at pre-grading only when a reviewer
# lists it in post-agent-classes.json as an EXPECTED-OUTPUT fixture -- a file
# the verifier compares the agent's own output against (sanitize-git-repo's
# sanitized ``process.py``) -- and cites the exact verifier lines that read it
# and do the comparison. Programs the verifier executes (schemelike's
# ``shadow_test/*.scm``) are never listed, so a leaked held-out test refuses.
# This only VALIDATES a reviewed entry; it never proposes one.


def expected_output_fixtures(task_dir: Path, entry: dict) -> dict[str, str]:
    """``tests relpath -> sha256`` of a reviewed entry, or ``Unsupported``.

    Each fixture must still hash to its frozen digest, and every citation
    must still be the exact line of the cited ``tests/`` file; the first
    citation must name the fixture. Drift in either means re-review.
    """
    tests = Path(task_dir) / "tests"
    out = {}
    for fixture in entry.get("expected_output_fixtures", []):
        rel, cites = fixture.get("path"), fixture.get("cites")
        path = tests / str(rel)
        if (not isinstance(rel, str) or ".." in Path(rel).parts or not path.is_file()
                or sha256(path) != fixture.get("sha256") or not fixture.get("justification")
                or not isinstance(cites, list) or len(cites) < 2):
            raise Unsupported(f"unreviewable expected-output fixture {rel!r}")
        for cite in cites:
            source = Path(task_dir) / str(cite.get("file", ""))
            if (not str(cite.get("file", "")).startswith("tests/") or not source.is_file()):
                raise Unsupported(f"{rel}: citation outside tests/")
            lines = source.read_text(errors="replace").splitlines()
            line = cite.get("line")
            if (not isinstance(line, int) or not 0 < line <= len(lines)
                    or lines[line - 1] != cite.get("text")):
                raise Unsupported(f"{rel}: cited line {cite.get('file')}:{line} changed")
        if Path(rel).name not in cites[0]["text"]:
            raise Unsupported(f"{rel}: first citation does not name the fixture")
        out[rel] = fixture["sha256"]
    return out


def main(argv: list[str]) -> int:
    import driver

    task_dir, image = Path(argv[0]), argv[1]
    cache = Path(argv[argv.index("--clone-cache") + 1]) if "--clone-cache" in argv else None
    derived = derive(task_dir, set(driver.gold_digests(task_dir)), cache)
    print(json.dumps({"task_payload_sha256": driver.task_payload(task_dir)[0],
                      "image_digest": image,
                      "dockerfile_sha256": sha256(task_dir / "environment/Dockerfile"),
                      **derived}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
