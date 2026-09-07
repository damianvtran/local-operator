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
- Prompt-identical bounded file contents collapse to one. Symlinks are never
  followed: automatic guidance cannot cross the repository's trust boundary.

Rendering contract: the files ride the byte-stable HEAD of the system prompt
(read once at session start, exactly like the operator's own instructions),
farthest-first so the nearest file — the most specific — lands last and is
read most prominently. Precedence is stated in the wrapper: repo guidance
describes the project's defaults; a direct instruction in the conversation
still wins, as it does over every standing instruction.

Large files ship as HEAD + SECTION INDEX, not as a byte-truncated dump
--------------------------------------------------------------------

A guidance file up to :data:`GUIDANCE_HEAD_BYTES` ships whole, unchanged. A
file larger than that ships as its first ``GUIDANCE_HEAD_BYTES`` (cut on a
LINE boundary) followed by a generated index of every remaining section with
its line range, plus an imperative to read that range before acting on its
subject.

This is a CORRECTNESS FIX before it is a token saving. The previous contract
kept the first 64KiB of each file and appended a one-line "truncated" note.
For this repository's own AGENTS.md — 93,355 bytes, 11 top-level sections —
that silently dropped FIVE whole sections from every session, chosen by byte
offset rather than by meaning: the timing/flake rules, the TUI widget
conventions, "Adding a configuration key", the analytics section and the
tool-surface footprint ladder. A rule past the cut was not merely absent, it
was *unfindable*: nothing in the prompt said it existed, so the model could
not know to go looking. The head/index form inverts that — every section is
named and addressable even though only the head is resident.

The pattern is deliberately the one the harness already ships for its own
procedures: ``<guides>`` lists guides "by name and description only — the
body loads on demand" and instructs the model that it MUST read the body
before acting. That works in practice, and it needs NO new resolver here: a
guidance file is a real path on disk and the ``read`` tool already takes a
path and a line range, so the index emits exactly what ``read`` consumes.

Why the head stays in this block rather than moving to a later, cheaper one:
this block is part of the cached prefix. Blocks after it are deliberately
kept breakpoint-free, so anything moved there is re-sent UNCACHED on every
turn — strictly worse than a one-time re-cache when the file changes.

The whole feature can be switched off with ``LOCAL_OPERATOR_CONTEXT_FILES=0``
when a directory's guidance files should not be trusted or the context budget
matters more than the conventions.
"""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

#: How many guidance files ride one system prompt. Nearest wins; deeper
#: ancestors beyond this are dropped rather than silently overflowing the
#: start-context budget (the 30k contract in docs/REWRITE.md).
MAX_CONTEXT_FILES = 5

#: Per-file ingest cap for the *resident* text and for the discovery digest.
#: A guidance file is instructions, not documentation; past this the file is
#: read on demand (it is on disk and grep-able) instead of occupying every
#: turn's cached prefix.
MAX_FILE_BYTES = 64 * 1024

#: How much of an oversized guidance file stays resident in the prompt.
#:
#: The trade-off is adherence against cost, and it is asymmetric. Rules the
#: agent breaks *without knowing it should have looked something up* — a
#: release gate, a review gate, "never symlink a venv", "read the committed
#: ref, not the working tree" — only work when they are resident; a head too
#: small silently converts those into rules nobody consults. Reference
#: material (timing analysis, widget conventions, subsystem internals) is safe
#: to leave lazy: the agent knows it is about to edit a widget, so an index
#: entry is enough to send it to the file.
#:
#: 8KiB was chosen over 6KiB and 12KiB against this repository's own 93KiB
#: AGENTS.md: it carries the environment/test/lint gates that apply to every
#: task regardless of subject, while cutting ~20.7k tokens from a fresh
#: session. Raising it buys progressively less — the material past 8KiB is
#: increasingly subject-specific, which is exactly the material an index
#: serves well. Lowering it starts evicting unconditional gates.
#:
#: NOTE this is a byte offset into the file as the operator wrote it. It is
#: NOT a claim that the first 8KiB of an arbitrary AGENTS.md are its most
#: important bytes — that is a property of how the file is ordered, which is
#: the file owner's business and not something this module edits or reorders.
GUIDANCE_HEAD_BYTES = 8 * 1024

#: Ceiling on the streaming scan that builds the section index. The scan
#: keeps only headings (a few hundred bytes) in memory, so this bounds I/O
#: rather than allocation, but a guidance file is not a corpus: past this the
#: index simply stops and says so.
MAX_SCAN_BYTES = 2 * 1024 * 1024

#: Filenames considered, in priority order per directory.
CANDIDATE_NAMES = ("AGENTS.md", "CLAUDE.md")


def _read_bounded(path: Path) -> tuple[bytes, bool]:
    """Read one regular file without following links or exceeding the cap.

    ``Path.is_symlink`` followed by ``open`` has a swap window. ``O_NOFOLLOW``
    makes the kernel enforce the trust boundary at the actual read, while the
    ``MAX_FILE_BYTES + 1`` probe determines truncation without ingesting the
    rest of an attacker-controlled file.
    """
    with _open_nofollow(path) as stream:
        probe = stream.read(MAX_FILE_BYTES + 1)
    return probe[:MAX_FILE_BYTES], len(probe) > MAX_FILE_BYTES


class _Section:
    """One markdown heading and the line span it owns.

    ``end`` is inclusive and is the line before the next heading of the same
    or higher level, so a section's span is exactly what ``read`` should be
    given to see that section and nothing else.
    """

    __slots__ = ("level", "title", "start", "end")

    def __init__(self, level: int, title: str, start: int) -> None:
        self.level = level
        self.title = title
        self.start = start
        self.end = start

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"_Section(L{self.start}-{self.end}, {'#' * self.level} {self.title})"


def _scan_sections(path: Path, max_level: int = 3) -> tuple[list[_Section], int, bool]:
    """Headings, total line count, and whether the scan hit its ceiling.

    Streams the file line by line: the index must describe a file far larger
    than the prompt can hold, so nothing but the headings is retained. Fenced
    code blocks are tracked because ``#`` starts a comment in most of the
    shell snippets these files carry, and a comment indexed as a section
    sends a later ``read`` to the wrong range.
    """
    sections: list[_Section] = []
    total_lines = 0
    scanned = 0
    truncated_scan = False
    fence: str | None = None

    with _open_nofollow(path) as stream:
        for raw in stream:
            scanned += len(raw)
            if scanned > MAX_SCAN_BYTES:
                truncated_scan = True
                break
            total_lines += 1
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            stripped = line.lstrip()
            # ``` or ~~~ toggles; the closing fence must match the opener so a
            # ```python block containing ``` in prose does not end it early.
            if stripped.startswith("```") or stripped.startswith("~~~"):
                marker = stripped[:3]
                if fence is None:
                    fence = marker
                elif fence == marker:
                    fence = None
                continue
            if fence is not None or not line.startswith("#"):
                continue
            level = len(line) - len(line.lstrip("#"))
            if level > max_level or not line[level:].startswith(" "):
                continue
            title = line[level:].strip()
            if not title:
                continue
            sections.append(_Section(level, title, total_lines))

    # A section's span ends where the next same-or-higher heading begins.
    for index, section in enumerate(sections):
        end = total_lines
        for later in sections[index + 1 :]:
            if later.level <= section.level:
                end = later.start - 1
                break
        section.end = end
    return sections, total_lines, truncated_scan


def _open_nofollow(path: Path):
    """``open`` that the kernel refuses to point at a symlink."""
    if path.is_symlink():
        raise OSError(f"refusing symlinked guidance: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise OSError(f"not a regular file: {path}")
    return os.fdopen(descriptor, "rb")


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
        if not candidate.is_symlink() and candidate.is_file():
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
                bounded, truncated = _read_bounded(found_here)
                digest_state = hashlib.sha256()
                digest_state.update(bounded)
                digest_state.update(b"\x01" if truncated else b"\x00")
                digest = digest_state.hexdigest()
            except OSError:
                digest = None  # unreadable/link/non-regular: never inject
                found_here = None
            if found_here is not None and digest is not None and digest not in seen_digests:
                seen_digests.add(digest)
                found.append(found_here)
        if directory == stop or directory == home or directory == directory.parent:
            break
    # Nearest-last above; the prompt wants farthest-first so the nearest file
    # (most specific) is the LAST thing in the block.
    found = found[:MAX_CONTEXT_FILES]  # keep nearest before reversing for prompt order
    found.reverse()
    return found


def _read_head(path: Path) -> tuple[str, int, bool]:
    """Leading text cut on a LINE boundary, its line count, and whether more
    of the file remains.

    Cutting on a line boundary rather than at the raw byte offset is the point:
    the previous contract sliced mid-sentence (and mid-word), so the last thing
    the model read was a fragment it could neither act on nor recognise as
    incomplete.
    """
    with _open_nofollow(path) as stream:
        probe = stream.read(GUIDANCE_HEAD_BYTES + 1)
    if len(probe) <= GUIDANCE_HEAD_BYTES:
        text = probe.decode("utf-8", errors="replace")
        return text, text.count("\n") + (0 if text.endswith("\n") or not text else 1), False
    head = probe[:GUIDANCE_HEAD_BYTES]
    cut = head.rfind(b"\n")
    if cut > 0:
        head = head[:cut]
    text = head.decode("utf-8", errors="replace")
    return text, text.count("\n") + 1, True


def _render_index(path: Path, shown: str, head_lines: int) -> str:
    """The section index that makes the non-resident remainder reachable."""
    try:
        sections, total_lines, scan_truncated = _scan_sections(path)
    except OSError:
        return ""
    # Level 1 is the document's title, not a section: it spans the whole file,
    # so an index row for it says nothing the path and line count do not. Its
    # span is still scanned, because an H2's range must end at the next H1 too.
    remaining = [s for s in sections if s.end > head_lines and s.level >= 2]
    if not remaining:
        return ""
    rows = []
    for section in remaining:
        # A section that STARTS inside the head is still listed: the model saw
        # its opening and not its rest, so the range offered is the remainder.
        start = max(section.start, head_lines + 1)
        indent = "  " * (section.level - 2)
        rows.append(f"{indent}- L{start}-{section.end}: {section.title}")
    listing = "\n".join(rows)
    note = (
        f"\n(index scan stopped at {MAX_SCAN_BYTES // 1024}KiB; later sections "
        "are not listed \u2014 read the file directly)"
        if scan_truncated
        else ""
    )
    # The imperative is deliberately as strong as the one <guides> carries,
    # and for the same reason: a section the model believes it already knows
    # is exactly the section whose local amendment it is about to violate.
    return (
        f"\nThe rest of this file is NOT included above. It is on disk at "
        f"`{shown}` ({total_lines} lines); these are its remaining sections "
        f"with the line ranges to read:\n\n"
        f"{listing}{note}\n\n"
        f"When a task touches any subject listed here, you MUST `read` that "
        f"line range of `{shown}` BEFORE acting or answering \u2014 even when you "
        f"believe you already know the answer, and even when the rules above "
        f"seem to cover it. These sections state this project's specific "
        f"gates, measured numbers and landmines; assuming the general case is "
        f"how an agent confidently does the wrong thing here."
    )


def render_context_files(files: list[Path], cwd: str | Path) -> str:
    """The ``<repo-guidance>`` block for the system prompt head.

    A file within :data:`GUIDANCE_HEAD_BYTES` ships whole with no index \u2014 a
    2KB AGENTS.md needs no ceremony, and an index of sections the model can
    already see is pure noise. Larger files ship head + index; see the module
    docstring for why that beats the byte-truncated dump it replaces.
    """
    if not files:
        return ""
    base = Path(cwd).resolve()
    parts: list[str] = []
    for path in files:
        try:
            text, head_lines, has_more = _read_head(path)
        except OSError:
            continue
        try:
            shown = str(path.relative_to(base))
        except ValueError:
            shown = str(path)
        body = text.strip()
        if has_more:
            body += "\n" + _render_index(path, shown, head_lines)
        parts.append(f'<file path="{shown}">\n{body}\n</file>')
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
