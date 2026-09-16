#!/usr/bin/env python3
"""Generate ``local_operator/agent_seeds/manifest.json`` from the packaged seeds.

The manifest is the ONE published list of built-in agent names (contract §5.2,
§5.4): it lives in this public repository, agent-server fetches it over HTTPS
and cites it in the refusal that stops somebody publishing an agent under a
name the app already ships. It is GENERATED and committed, never hand-written.
``tests/unit/test_agent_seed_manifest.py`` re-renders it and requires
byte-identical output, so a seventh seed — or an edit to one of the six — cannot
ship a manifest that still describes the previous set. That check is the only
thing keeping the two name lists in step (contract §7.3 risk 5), which makes two
properties below load-bearing rather than stylistic:

**No implicit provenance reads.** ``generated_at`` and ``generator_version``
are provenance, and regeneration with no arguments PRESERVES the committed
values; only an explicit ``--generated-at`` / ``--generator-version`` (or
``--stamp``) replaces them. Reading the clock or the installed package version
implicitly would move the output of every regeneration, and the release flow
bumps that version in a pyproject-only PR, so the drift check would go red after
every release and stay red until somebody regenerated it — a check that is
routinely red is a check nobody reads. The same reasoning is why each seed
carries its OWN explicit ``version:`` frontmatter rather than letting
``version`` default to the package version: a seed's version is a fact about the
seed, not about the release it happened to ship in.

**Stable ordering.** ``list_seeds()`` is sorted, seeds are re-sorted here, and
JSON object keys are emitted in insertion order, so identical seeds always
render identical bytes.

The seeds directory is the right home (contrary to e.g. ``docs/``) because
``pyproject.toml``'s package-data list is explicit and does not recurse: the
manifest ships in the wheel only because ``"agent_seeds/*.json"`` is listed
beside ``"agent_seeds/*.md"``. A manifest that does not ship is a silent
failure — the installed package would carry the seeds with no record of what
they hash to. ``--check`` fails when the committed file is stale, which is what
a CI job or a reviewer runs instead of trusting the diff.

Run:
    .venv/bin/python scripts/gen_agent_seed_manifest.py
    .venv/bin/python scripts/gen_agent_seed_manifest.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
# Scripts under scripts/ read the tree they live in regardless of which venv
# launched them (a console script's sys.path[0] is .venv/bin, not the cwd), so
# the generator always describes this checkout's seeds.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from local_operator.agent_profiles import (  # noqa: E402
    MAX_INSTRUCTIONS_CHARS,
    ROLE_TAG,
    SEEDS_DIR,
    _split_frontmatter,
    list_seeds,
    load_seed,
    seed_tags,
)

#: Bumped when the manifest's own shape changes; agent-server refuses any other
#: value rather than guessing at a schema it does not know (contract §5.3).
SCHEMA_VERSION = 1

#: The project the seeds belong to, quoted in the manifest so a consumer can
#: tell where a cached copy came from without a second request.
SOURCE = "https://github.com/damianvtran/local-operator"

#: Base of the per-seed ``source_url``. This is the exact prefix agent-server is
#: told to fetch (contract §5.2), and it is what the ``name_reserved_builtin``
#: refusal links to, so it must name the same branch the fetch reads: ``main``.
RAW_BASE = "https://raw.githubusercontent.com/damianvtran/local-operator/main"

#: Where the generator writes, and where the loader's package data puts it.
MANIFEST_PATH = SEEDS_DIR / "manifest.json"


def _frontmatter_version(name: str, text: str) -> str:
    """The seed's own ``version:`` frontmatter, which must be explicit.

    Parsed with :func:`_split_frontmatter` — the same parser the loader uses —
    rather than a second YAML reader beside it, because two parsers is how the
    two would later disagree about the same bytes (contract §5.2). A missing or
    blank value is a hard error rather than a default: the package version
    would move this field on every release bump, and a moved field makes the
    byte-identity check red for a change that touched no seed.
    """

    meta, _body = _split_frontmatter(text)
    version = str(meta.get("version") or "").strip()
    if not version:
        raise SystemExit(
            f"{name}.md has no 'version:' frontmatter. Every packaged seed must "
            "declare its own version: the manifest is checked for byte-identity, "
            "and a version that follows the package would change on every release."
        )
    return version


def _entry(name: str) -> dict[str, Any]:
    """Build one manifest entry from the seed the app actually installs."""

    profile = load_seed(name)
    if profile is None:
        raise SystemExit(f"seed {name!r} is in the catalogue but does not load")
    if profile.name != name:
        # ``list_seeds()`` is keyed by FILENAME and the loader takes the
        # profile's name from the ``name:`` frontmatter, so the two can drift.
        # The manifest's ``name`` has to be the name the app would INSTALL —
        # that is the string a published agent collides with — so a mismatch
        # must fail here rather than publish a name nobody can install.
        raise SystemExit(
            f"{name}.md declares name: {profile.name!r}; the seed catalogue is keyed by "
            f"filename, so rename the file or the frontmatter to make them agree."
        )
    text = (SEEDS_DIR / f"{name}.md").read_text(encoding="utf-8", errors="replace")
    _meta, body = _split_frontmatter(text)
    if len(body) > MAX_INSTRUCTIONS_CHARS:
        # The manifest hashes ONE definition of the body. A body long enough to
        # be truncated by the loader is a seed that silently ships less guidance
        # than it is written with, so refuse rather than describe a shortened
        # hash that no reader could reproduce from the file.
        raise SystemExit(
            f"{name}.md is {len(body)} characters; the loader keeps "
            f"{MAX_INSTRUCTIONS_CHARS}, so the manifest's hash would describe a "
            "body the seed does not contain."
        )

    # ``profile.instructions`` is the exact string the loader holds and
    # ``install_seed`` writes, so hashing it cannot describe a body the app
    # would not run. What is published is a description of the INSTALLED role,
    # which is also why tags come from ``seed_tags`` (the encoding install
    # writes) and the category is ``ROLE_TAG`` (what install sets).
    instructions = profile.instructions
    return {
        "name": name,
        "description": profile.description,
        "when_to_use": profile.when_to_use,
        "version": _frontmatter_version(name, text),
        "instructions_sha256": hashlib.sha256(instructions.encode("utf-8")).hexdigest(),
        "instructions_chars": len(instructions),
        "tools": list(profile.tools) if profile.tools is not None else None,
        "effort": profile.effort,
        "delegate": bool(profile.may_delegate),
        "tags": list(seed_tags(profile)),
        "categories": [ROLE_TAG],
        "source_url": f"{RAW_BASE}/local_operator/agent_seeds/{name}.md",
    }


def _committed_provenance(manifest_path: Path) -> dict[str, Any]:
    """The provenance already recorded in the committed manifest, if readable.

    Unreadable or malformed means "no committed value" rather than an error:
    the generator has to be able to write a first manifest, and a hand-mangled
    one is repaired by regenerating it, which is also what the drift test tells
    the author to do.
    """

    try:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _resolve_provenance(
    manifest_path: Path,
    *,
    generated_at: str | None,
    generator_version: str | None,
) -> tuple[str, str]:
    """Resolve ``(generated_at, generator_version)`` without touching a clock
    unless there is nothing to preserve (see the module docstring)."""

    if generated_at and generator_version:
        return generated_at, generator_version

    committed = _committed_provenance(manifest_path)
    resolved_at = generated_at or str(committed.get("generated_at") or "")
    resolved_version = generator_version or str(committed.get("generator_version") or "")
    if not resolved_at:
        # First generation, or the value was lost: the clock is read only here,
        # and a caller pinning SOURCE_DATE_EPOCH (the reproducible-build
        # convention) gets a reproducible value even on this branch.
        epoch = os.environ.get("SOURCE_DATE_EPOCH")
        stamp = (
            datetime.fromtimestamp(int(epoch), tz=timezone.utc)
            if epoch
            else datetime.now(timezone.utc)
        )
        resolved_at = stamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    if not resolved_version:
        try:
            resolved_version = version("local-operator")
        except PackageNotFoundError:  # pragma: no cover - running from a source tree
            resolved_version = "0.0.0"
    return resolved_at, resolved_version


def render(
    manifest_path: Path = MANIFEST_PATH,
    *,
    generated_at: str | None = None,
    generator_version: str | None = None,
) -> str:
    """Render the manifest as the exact bytes that belong in the repository."""

    stamp, version = _resolve_provenance(
        manifest_path, generated_at=generated_at, generator_version=generator_version
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": stamp,
        "generator_version": version,
        "source": SOURCE,
        "seeds": [_entry(name) for name in sorted(list_seeds())],
    }
    # ``indent=2`` and a trailing newline: the file is read by humans in review
    # and by agents' diffs, and a JSON document with no final newline is a
    # gratuitous "\ No newline at end of file" in every diff it appears in.
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the packaged built-in agent manifest from the seeds."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 when the committed manifest differs; write nothing.",
    )
    parser.add_argument("--generated-at", default=None, help="ISO-8601 Z stamp to record.")
    parser.add_argument(
        "--generator-version", default=None, help="Package version to record as generator_version."
    )
    parser.add_argument(
        "--out", type=Path, default=MANIFEST_PATH, help="Manifest path (default: packaged path)."
    )
    args = parser.parse_args(argv)

    rendered = render(
        args.out, generated_at=args.generated_at, generator_version=args.generator_version
    )

    if args.check:
        try:
            committed = args.out.read_text(encoding="utf-8")
        except OSError:
            print(f"{args.out} is missing; run this script without --check", file=sys.stderr)
            return 1
        if committed != rendered:
            print(
                f"{args.out} is stale: regenerate it with "
                "`.venv/bin/python scripts/gen_agent_seed_manifest.py`",
                file=sys.stderr,
            )
            return 1
        print(f"{args.out} matches the packaged seeds")
        return 0

    args.out.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.out} ({len(json.loads(rendered)['seeds'])} seeds)")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
