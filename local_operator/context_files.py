"""Repository guidance auto-discovery (AGENTS.md / CLAUDE.md).

Most repositories the agent works in carry a standing instructions file —
AGENTS.md increasingly, CLAUDE.md historically — that states the project's
conventions, gates, and landmines. Without this module the agent works blind
to all of it: the only standing instructions it saw were the operator's
machine-wide ``system_prompt.md``, which is about the MACHINE, not the repo
the prompt happens to run in.

Discovery contract:

- Walk UP from the session's working directory, one directory at a time.
- In each directory accept ``AGENTS.md``; when absent, ``CLAUDE.md`` stands
  in (never both — the two names are two covers for one intent).
- Stop at the git repository root (inclusive) when one is found: guidance
  above the repo belongs to the enclosing workspace, and walking past the
  root into the home directory would pick up unrelated projects' files. With
  no git root, stop at the user's home directory (inclusive) or the
  filesystem root, whichever comes first.
- Keep at most :data:`MAX_CONTEXT_FILES` files, the NEAREST ones — a deep
  monorepo can nest more levels than the prompt should carry, and the
  nearest files are the most specific.
- Byte-identical files collapse to one (symlinked or vendored duplicates).

Rendering contract: the files ride the byte-stable HEAD of the system prompt
(read once at session start, exactly like the operator's own instructions),
farthest-first so the nearest file — the most specific — lands last and is
read most prominently. Precedence is stated in the wrapper: repo guidance
describes the project's defaults; a direct instruction in the conversation
still wins, as it does over every standing instruction.

The whole feature can be switched off with ``LOCAL_OPERATOR_CONTEXT_FILES=0``
when a directory's guidance files should not be trusted or the context budget
matters more than the conventions.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

#: How many guidance files ride one system prompt. Nearest wins; deeper
#: ancestors beyond this are dropped rather than silently overflowing the
#: start-context budget (the 30k contract in docs/REWRITE.md).
MAX_CONTEXT_FILES = 5

#: Per-file byte cap. A guidance file is instructions, not documentation;
#: past this the file is read on demand (it is on disk and grep-able) instead
#: of occupying every turn's cached prefix.
MAX_FILE_BYTES = 64 * 1024

#: Filenames considered, in priority order per directory.
CANDIDATE_NAMES = ("AGENTS.md", "CLAUDE.md")


def _git_root(start: Path) -> Path | None:
    """Nearest enclosing directory containing ``.git`` (file or dir — both
    worktrees and submodules use the file form), or None."""
    for directory in (start, *start.parents):
        if (directory / ".git").exists():
            return directory
    return None


def _file_for(directory: Path) -> Path | None:
    for name in CANDIDATE_NAMES:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def discover_context_files(cwd: str | Path) -> list[Path]:
    """Ancestor guidance files, farthest-first. Empty when disabled or none."""
    if os.environ.get("LOCAL_OPERATOR_CONTEXT_FILES", "1").strip() in ("0", "false", "no"):
        return []
    start = Path(cwd).resolve()
    home = Path.home()
    stop = _git_root(start)
    found: list[Path] = []
    seen_digests: set[str] = set()
    for directory in (start, *start.parents):
        found_here = _file_for(directory)
        if found_here is not None:
            try:
                digest = hashlib.sha256(found_here.read_bytes()).hexdigest()
            except OSError:
                digest = None  # unreadable: skip rather than partially trust
                found_here = None
            if found_here is not None and digest is not None and digest not in seen_digests:
                seen_digests.add(digest)
                found.append(found_here)
        if directory == stop or directory == home or directory == directory.parent:
            break
    # Nearest-last above; the prompt wants farthest-first so the nearest file
    # (most specific) is the LAST thing in the block.
    found.reverse()
    return found[:MAX_CONTEXT_FILES]


def render_context_files(files: list[Path], cwd: str | Path) -> str:
    """The ``<repo-guidance>`` block for the system prompt head."""
    if not files:
        return ""
    base = Path(cwd).resolve()
    parts: list[str] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(text.encode("utf-8")) > MAX_FILE_BYTES:
            text = text[:MAX_FILE_BYTES] + "\n[...truncated at 64KiB; read the file for the rest]"
        try:
            shown = str(path.relative_to(base))
        except ValueError:
            shown = str(path)
        parts.append(f'<file path="{shown}">\n{text.strip()}\n</file>')
    if not parts:
        return ""
    return (
        "## Repository guidance\n\n"
        "These instruction files were found in this directory's ancestors and "
        "state the project's conventions. Treat them as the project's "
        "defaults; a direct instruction in the conversation still wins.\n\n"
        "<repo-guidance>\n" + "\n".join(parts) + "\n</repo-guidance>"
    )


def load_repo_guidance(cwd: str | Path) -> str:
    """One call: discover + render. ``""`` when there is nothing to inject."""
    return render_context_files(discover_context_files(cwd), cwd)
