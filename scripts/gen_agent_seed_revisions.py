#!/usr/bin/env python3
"""Generate ``local_operator/agent_seeds/seed_revisions.json`` from git history.

The ledger is the proof behind #2060's startup update pass: every text a
built-in seed ever SHIPPED, oldest first per seed, so a row whose canonical
vector equals an entry is provably an unedited published revision and the
entry's POSITION is what decides "behind" from "ahead" (version strings repeat
— aida shipped four ``1.0.0`` builds, two of them same-day — so they cannot).
It is generated and committed, never hand-written, and the runtime only ever
reads it.

THREE MODES, and the discipline each one keeps:

* **Normal** (no arguments): preserve every committed entry, and APPEND the
  packaged revision for a seed whose full vector (class included) differs from
  its tail. Never rewrites, never removes, never reads git HISTORY — a
  no-op when the tails already match, which is the state the byte-identity
  test asserts and the reason CI can check the file on a shallow clone. An
  append must be attributable to a commit: the seed file has to be committed
  (its bytes equal its HEAD blob) so the new entry can name the commit that
  introduced it, and an uncommitted or unattributable seed REFUSES with the
  remedy rather than recording a made-up sha. This is the mode the release
  flow runs after a seed edit is committed.
* **``--check``**: validate the committed file in place — schema, per-seed
  order/uniqueness, and every seed's tail equal to the packaged starter — and
  write NOTHING. No git, no history: safe on CI's fetch-depth-2 checkout,
  where the byte-identity test's normal-mode render is also a no-op. Exit 1 on
  any violation.
* **``--bootstrap-from-git``**: the one-time enumeration that built the first
  file: every commit touching any seed (``git log --follow``), deduped by the
  full canonical vector per seed, sha = the commit that FIRST carried the
  text. Refuses when ``git rev-parse --is-shallow-repository`` is true — and
  writes NOTHING on a refusal — unless ``--allow-shallow`` is passed, which
  is ONLY for a clone that is marked shallow yet carries every seed's
  history (this fleet's reference clone; the per-seed check below is what
  makes the override safe). Independently, a bootstrap never DECREASES a
  target's revision count vs the file already there: a walk that would
  truncate the committed ledger refuses instead, whichever door it came
  through (QA round 1, Q3 / agent review round 1, R1-4). Re-run only when
  bootstrapping; the normal mode's append rule keeps the file current
  afterwards.

SEED-EDIT WORKFLOW (the sequence a seed editor must follow, because a ledger
entry names the commit that first carried the text — agent review round 1,
R1-10):

1. Edit ``local_operator/agent_seeds/<name>.md`` (bump ``version`` if the
   move is user-visible).
2. COMMIT the seed file. The normal mode refuses an uncommitted seed: the
   new entry's sha has to name a real commit, and a dirty file has none.
3. Run the normal mode: ``.venv/bin/python scripts/gen_agent_seed_revisions.py``
   — it appends the packaged revision and rewrites nothing else.
4. Commit the regenerated ``seed_revisions.json`` in the same PR. The
   byte-identity unit test fails otherwise, naming this command.

Run:
    .venv/bin/python scripts/gen_agent_seed_revisions.py            # append if moved
    .venv/bin/python scripts/gen_agent_seed_revisions.py --check    # CI / reviewer
    .venv/bin/python scripts/gen_agent_seed_revisions.py --bootstrap-from-git
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
# Scripts under scripts/ read the tree they live in regardless of which venv
# launched them (a console script's sys.path[0] is .venv/bin, not the cwd), so
# the generator always describes this checkout's seeds.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import local_operator.agent_profiles as agent_profiles  # noqa: E402
from local_operator.agent_profiles import (  # noqa: E402
    SEED_REVISIONS_NAME,
    SEED_REVISIONS_SCHEMA_VERSION,
    SeedRevision,
    _split_frontmatter,
    list_seeds,
    load_seed,
    load_seed_class,
    make_seed_revision,
    render_seed_revisions,
)

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

#: The path the ledger is committed at, relative to the repository. Normal-mode
#: appends resolve the introducing commit against this path with ``git show``,
#: so it must be the committed spelling (the same one ``git log`` sees).
LEDGER_REL_PATH = f"local_operator/agent_seeds/{SEED_REVISIONS_NAME}"


def _seeds_dir(seeds_dir: Path | None) -> Path:
    """The seed directory to describe, resolved at CALL time.

    ``agent_profiles.SEEDS_DIR`` is a module global precisely so tests can
    point it at a scratch copy; reading it here (rather than importing the
    value) keeps the generator honest against that seam.
    """

    return Path(seeds_dir) if seeds_dir is not None else agent_profiles.SEEDS_DIR


def _declared_class(text: str) -> str | None:
    """A seed file's declared ``class:`` frontmatter, or None when absent.

    The same reading :func:`agent_profiles.load_seed_class` performs for the
    packaged tree, but over arbitrary text — the bootstrap reads a seed at a
    COMMIT, not at HEAD.
    """

    meta, _body = _split_frontmatter(text)
    raw = meta.get("class")
    if raw is None or not str(raw).strip():
        return None
    return agent_profiles.normalize_action_class(raw)


def _entry_full_vector(entry: SeedRevision) -> tuple[Any, ...]:
    """Class included: the vector the append rule keys on."""

    return (
        entry.instructions_sha256,
        entry.description,
        entry.tools,
        entry.effort,
        entry.delegate,
        entry.action_class,
    )


def _profile_full_vector(profile: Any, declared_class: str | None) -> tuple[Any, ...]:
    """The packaged seed's full vector, shaped as :func:`_entry_full_vector`."""

    vector = agent_profiles.seed_revision_vector(profile)
    return (
        hashlib.sha256(vector[0].encode("utf-8")).hexdigest(),
        vector[1],
        vector[2],
        vector[3],
        vector[4],
        declared_class,
    )


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _git(*args: str) -> subprocess.CompletedProcess[bytes] | None:
    """Run git in the repository; None when git itself is unavailable."""

    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=False)
    except OSError:  # pragma: no cover - no git on PATH
        return None


def _introducing_sha(seed_path: Path) -> str | None:
    """The commit that introduced the seed's CURRENT text, or None.

    ``None`` — and the append refusal that follows — covers every case where
    the attribution would be a guess: git unavailable, a shallow clone, no
    commit yet, or a working tree whose bytes differ from the HEAD blob (an
    uncommitted edit). The ledger's honesty rule is that every entry names the
    commit that introduced its text; a guessed sha would poison the deep
    verification the whole file rests on.
    """

    relative = seed_path.relative_to(REPO).as_posix()
    head = _git("log", "-1", "--format=%H", "--", relative)
    if head is None or head.returncode != 0:
        return None
    sha = head.stdout.decode("utf-8", "replace").strip()
    if not _SHA_RE.match(sha):
        return None
    blob = _git("show", f"{sha}:{relative}")
    if blob is None or blob.returncode != 0:
        return None
    if blob.stdout != seed_path.read_bytes():
        return None
    return sha


def render(*, seeds_dir: Path | None = None) -> str:
    """Render the ledger as the exact bytes that belong in the repository.

    Committed entries are preserved verbatim (append-only); for every packaged
    seed whose full vector differs from its tail, a new entry is appended. An
    append that cannot name its introducing commit refuses with a SystemExit
    naming the remedy — never a placeholder sha. The entries are read through
    the runtime loader, so this describes the SAME file the app will read.
    """

    directory = _seeds_dir(seeds_dir)
    committed = agent_profiles.load_seed_revisions(directory)
    payload: dict[str, list[SeedRevision]] = {
        name: list(committed.get(name, ())) for name in sorted(committed)
    }

    for name in sorted(list_seeds()):
        profile = load_seed(name)
        if profile is None:
            raise SystemExit(f"seed {name!r} is in the catalogue but does not load")
        declared = load_seed_class(name)
        entries = payload.setdefault(name, [])
        if entries and _entry_full_vector(entries[-1]) == _profile_full_vector(profile, declared):
            continue
        seed_path = directory / f"{name}.md"
        sha = _introducing_sha(seed_path)
        if sha is None:
            raise SystemExit(
                f"{seed_path} changed but cannot be attributed to a commit; commit the "
                "seed edit first (a full, non-shallow clone is required), then re-run "
                "this generator so the ledger can name the commit that introduced the text."
            )
        version = agent_profiles.load_seed_version(name)
        entries.append(
            make_seed_revision(profile, sha=sha, version=version, declared_class=declared)
        )

    return render_seed_revisions({name: tuple(entries) for name, entries in payload.items()})


def _entry_schema_errors(seed: str, index: int, row: object) -> list[str]:
    """Field-level violations of one ledger entry; empty means valid."""

    errors: list[str] = []
    if not isinstance(row, dict):
        return [f"{seed}[{index}]: entry is not an object"]
    sha = row.get("sha")
    if not isinstance(sha, str) or not _SHA_RE.match(sha):
        errors.append(f"{seed}[{index}]: sha {sha!r} is not a 40-hex commit")
    version = row.get("version")
    if not isinstance(version, str):
        errors.append(f"{seed}[{index}]: version must be a string")
    digest = row.get("instructions_sha256")
    if not isinstance(digest, str) or not agent_profiles.is_sha256_hex(digest):
        errors.append(f"{seed}[{index}]: instructions_sha256 {digest!r} is not a sha256")
    if not isinstance(row.get("description"), str):
        errors.append(f"{seed}[{index}]: description must be a string")
    tools = row.get("tools")
    if tools is not None and not (
        isinstance(tools, list) and all(isinstance(item, str) for item in tools)
    ):
        errors.append(f"{seed}[{index}]: tools must be a list of strings or null")
    if row.get("effort") is not None and not isinstance(row.get("effort"), str):
        errors.append(f"{seed}[{index}]: effort must be a string or null")
    if not isinstance(row.get("delegate"), bool):
        errors.append(f"{seed}[{index}]: delegate must be a boolean")
    action_class = row.get("class")
    if action_class is not None and action_class not in ("reactive", "proactive"):
        errors.append(f"{seed}[{index}]: class must be 'reactive', 'proactive' or null")
    return errors


def check(ledger_path: Path | None = None, *, seeds_dir: Path | None = None) -> list[str]:
    """Validate the committed ledger IN PLACE; returns every violation.

    No git, no writes — this is what CI (a shallow checkout) and a reviewer
    run. The checks are the properties the runtime trusts: the schema parses,
    each seed's entries are strictly unique in append order, and every
    packaged seed's TAIL equals the packaged starter (the runtime's guard
    that draft texts in worktrees are never auto-applied).
    """

    directory = _seeds_dir(seeds_dir)
    target = Path(ledger_path) if ledger_path is not None else directory / SEED_REVISIONS_NAME
    errors: list[str] = []
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except OSError:
        return [f"{target} is missing; run this script without --check"]
    except ValueError as exc:
        return [f"{target} is not valid JSON: {exc}"]
    if not isinstance(payload, dict):
        return [f"{target}: top level must be an object"]
    if payload.get("schema_version") != SEED_REVISIONS_SCHEMA_VERSION:
        errors.append(
            f"schema_version {payload.get('schema_version')!r} != {SEED_REVISIONS_SCHEMA_VERSION}"
        )
    seeds = payload.get("seeds")
    if not isinstance(seeds, dict):
        return [*errors, "seeds must be an object keyed by seed name"]

    parsed: dict[str, tuple[SeedRevision, ...]] = {}
    for name, rows in seeds.items():
        if not isinstance(rows, list) or not rows:
            errors.append(f"{name}: no entries")
            continue
        seen: set[tuple[Any, ...]] = set()
        entries: list[SeedRevision] = []
        for index, row in enumerate(rows):
            errors.extend(_entry_schema_errors(str(name), index, row))
            if not isinstance(row, dict):
                continue
            vector = (
                row.get("instructions_sha256"),
                row.get("description"),
                tuple(row["tools"]) if isinstance(row.get("tools"), list) else None,
                row.get("effort"),
                row.get("delegate"),
                row.get("class"),
            )
            if vector in seen:
                errors.append(f"{name}[{index}]: repeats an earlier revision (append-only)")
            seen.add(vector)
            # The tail check below must read the file UNDER CHECK, not the
            # module's copy: a stale --out file whose tail drifted is exactly
            # the failure this gate exists to catch, and loading the module's
            # ledger here would validate the wrong bytes (found by this gate's
            # own test, which mutated a copy and watched it pass).
            entries.append(
                SeedRevision(
                    sha=str(row.get("sha") or ""),
                    version=str(row.get("version") or ""),
                    instructions_sha256=str(row.get("instructions_sha256") or ""),
                    description=str(row.get("description") or ""),
                    tools=tuple(row["tools"]) if isinstance(row.get("tools"), list) else None,
                    effort=str(row["effort"]) if row.get("effort") else None,
                    delegate=bool(row.get("delegate")),
                    action_class=(str(row["class"]) if row.get("class") else None),
                )
            )
        parsed[str(name)] = tuple(entries)

    for name in sorted(list_seeds()):
        seed = load_seed(name)
        if seed is None:  # pragma: no cover - guarded by the render path
            continue
        if name not in parsed:
            errors.append(f"{name}: missing from the ledger; regenerate it")
            continue
        if agent_profiles._packaged_tail_entry(name, parsed) is None:
            errors.append(
                f"{name}: the ledger's tail does not match the packaged starter; regenerate it"
            )
    return errors


def _history_available(relative_path: str) -> bool:
    """Whether ``relative_path``'s FULL history is present in this clone.

    The bootstrap's real requirement, checked directly instead of through
    ``--is-shallow-repository``: this fleet's reference clone is MARKED
    shallow (its initial fetch used a depth) while carrying every commit the
    seeds need, so a strict shallow refusal told an operator with the full
    relevant history to unshallow a repository that would enumerate
    identically. What actually matters is that the file's ADDING commit is
    reachable: with it, ``--follow`` sees the file's whole life; without it,
    the walk is truncated at a graft point and older revisions would be
    silently omitted from the ledger - the one failure a bootstrap must never
    have.
    """

    added = _git("log", "--format=%H", "--diff-filter=A", "--", relative_path)
    return added is not None and added.returncode == 0 and bool(added.stdout.strip())


def _shallow_repository() -> bool:
    """Whether git MARKS this repository shallow (``rev-parse --is-shallow-repository``).

    The bootstrap's outer guard, and deliberately COARSER than
    :func:`_history_available`: a marked-shallow clone may still carry every
    commit the seeds need (the fleet's reference clone does), which is why
    the override exists — but a bootstrap must be asked EXPLICITLY before it
    trusts a graft-marked clone, because the failure it would hide (a walk
    truncated at the graft point, silently omitting older revisions) leaves
    no trace in the ledger it writes. ``--is-shallow-repository`` is answered
    locally by git and needs no history fetch itself.
    """

    result = _git("rev-parse", "--is-shallow-repository")
    return result is not None and result.stdout.decode("utf-8", "replace").strip() == "true"


def _partition_entry_count(rendered: str) -> int:
    """Total revisions in a rendered ledger; 0 for anything unparseable."""

    try:
        payload = json.loads(rendered)
        return sum(len(rows) for rows in payload["seeds"].values())
    except (ValueError, KeyError, AttributeError, TypeError):
        return 0


def bootstrap_from_git(*, allow_shallow: bool = False) -> str:
    """Rebuild the whole ledger from repository history (one-time).

    Walks every commit that touched each seed (``--follow`` included),
    extracts the seed's canonical vector at that commit, keeps the FIRST
    commit to carry each full vector, and renders the result. This is the
    only mode that reads history.

    Two guards, because a truncated walk writes a plausible-looking ledger:
    a clone git MARKS shallow refuses outright (``--allow-shallow`` overrides
    for the known-complete marked clone), and every seed's ADD commit must be
    reachable. The caller additionally refuses to shrink an existing target
    (see ``main``); together these are the QA-round-1 Q3 answer, whose repro
    showed a depth-1 clone silently truncating 65 revisions to 10 with
    ``--check`` still passing afterwards.
    """

    if _shallow_repository() and not allow_shallow:
        raise SystemExit(
            "bootstrap: this repository is marked shallow; the history walk would "
            "silently omit older revisions. Re-run with --allow-shallow ONLY if the "
            "clone is known-complete (every seed's whole history present) despite "
            "the marker, or fetch the full history first."
        )

    for name in sorted(list_seeds()):
        relative = f"local_operator/agent_seeds/{name}.md"
        if not _history_available(relative):
            raise SystemExit(
                f"the history for {relative} is incomplete in this clone (a shallow "
                "fetch would truncate the walk); run `git fetch --unshallow` and re-run."
            )
    result: dict[str, tuple[SeedRevision, ...]] = {}
    scanned = 0
    for name in sorted(list_seeds()):
        relative = f"local_operator/agent_seeds/{name}.md"
        log = _git("log", "--follow", "--format=%H", "--reverse", "--", relative)
        if log is None or log.returncode != 0:
            raise SystemExit(f"git log failed for {relative}")
        entries: list[SeedRevision] = []
        seen: set[tuple[Any, ...]] = set()
        for sha in log.stdout.decode("utf-8", "replace").split():
            scanned += 1
            blob = _git("show", f"{sha}:{relative}")
            if blob is None or blob.returncode != 0:
                continue
            text = blob.stdout.decode("utf-8", "replace")
            profile = agent_profiles._profile_from_text(name, text)
            version = str(_split_frontmatter(text)[0].get("version") or "")
            entry = make_seed_revision(
                profile, sha=sha, version=version, declared_class=_declared_class(text)
            )
            vector = _entry_full_vector(entry)
            if vector in seen:
                continue
            seen.add(vector)
            entries.append(entry)
        if entries:
            result[name] = tuple(entries)
    # The walk's own count is the bootstrap's audit trail: the manager's
    # measured fleet state is 81 HEAD-reachable commits (96 with ``--all``,
    # which adds unmerged branches) -> 65 revisions, and a future re-bootstrap
    # should reproduce BOTH of those numbers (a shallower walk that still
    # produces plausible output is exactly the silent truncation the
    # shallow/marker guards and the caller's count check exist to stop; this
    # line makes it visible either way).
    print(f"scanned {scanned} commits touched seed files", file=sys.stderr)
    return render_seed_revisions(result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the packaged seed revision ledger from history."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the committed ledger in place; write nothing; exit 1 on violations.",
    )
    parser.add_argument(
        "--bootstrap-from-git",
        action="store_true",
        help=(
            "Rebuild the whole ledger from git history (one-time; refuses on a "
            "repository git marks shallow unless --allow-shallow is also given)."
        ),
    )
    parser.add_argument(
        "--allow-shallow",
        action="store_true",
        help=(
            "Permit --bootstrap-from-git on a marked-shallow clone that is KNOWN to "
            "carry every seed's history (the fleet's reference clone). A bootstrap "
            "that would still shrink the target's revision count refuses regardless."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Ledger path (default: the packaged path).",
    )
    args = parser.parse_args(argv)

    target = args.out if args.out is not None else agent_profiles.SEEDS_DIR / SEED_REVISIONS_NAME

    if args.check:
        errors = check(args.out)
        if errors:
            for line in errors:
                print(f"error: {line}", file=sys.stderr)
            return 1
        print(f"{target} matches the packaged seeds")
        return 0

    if args.bootstrap_from_git:
        rendered = bootstrap_from_git(allow_shallow=bool(args.allow_shallow))
        # NEVER TRUNCATE A TARGET: a walk that would leave FEWER revisions
        # than the file already carries has gone wrong somewhere (a mode-3
        # run against a thinner clone), and the file it would overwrite is
        # the committed ledger. Refuse, write nothing.
        existing = target.read_text(encoding="utf-8") if target.is_file() else ""
        existing_total = _partition_entry_count(existing)
        payload = json.loads(rendered)
        total = sum(len(rows) for rows in payload["seeds"].values())
        if existing and total < existing_total:
            raise SystemExit(
                f"bootstrap: the walk found {total} revisions but {target} already "
                f"carries {existing_total}; refusing to truncate the ledger "
                "(writes nothing)."
            )
        target.write_text(rendered, encoding="utf-8")
        print(f"wrote {target} (bootstrap: {total} revisions across {len(payload['seeds'])} seeds)")
        return 0

    rendered = render()
    committed = target.read_text(encoding="utf-8") if target.is_file() else ""
    if rendered == committed:
        print(f"{target} is unchanged ({len(json.loads(rendered)['seeds'])} seeds)")
        return 0
    target.write_text(rendered, encoding="utf-8")
    payload = json.loads(rendered)
    total = sum(len(rows) for rows in payload["seeds"].values())
    print(f"wrote {target} ({total} revisions across {len(payload['seeds'])} seeds)")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
