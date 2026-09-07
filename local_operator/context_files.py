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

THE ONE INVARIANT: an oversized file NEVER renders without disclosure. Some
files yield no listable sections at all — no headings, only H1s, or every
heading inside the head — and for those the index degrades to a bare pointer
(path, line count, the range to read) rather than to silence. Emitting a bare
head with nothing saying the file continues would drop the operator's rules
both invisibly and unrecoverably, which is worse than the byte cut this
replaces: that at least admitted it had truncated.

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
from typing import BinaryIO

#: How many guidance files ride one system prompt. Nearest wins; deeper
#: ancestors beyond this are dropped rather than silently overflowing the
#: start-context budget (the 30k contract in docs/REWRITE.md).
MAX_CONTEXT_FILES = 5

#: Bytes hashed when deciding whether two discovered files are the same file.
#: This does NOT bound the resident text -- :data:`GUIDANCE_HEAD_BYTES` does.
#: It is only the dedup probe, kept bounded so a nested tree of large guidance
#: files cannot be made to read gigabytes during discovery.
#:
#: Consequence worth knowing: two files identical in their first 64KiB but
#: differing afterwards collapse to one. That is pre-existing behaviour; the
#: file's length is folded into the digest so the common "same prefix,
#: different length" case stays distinguishable.
MAX_DIGEST_BYTES = 64 * 1024

#: How much of an oversized guidance file stays resident in the prompt.
#:
#: The trade-off is adherence against cost, and it is asymmetric in PRINCIPLE:
#: rules the agent breaks *without knowing it should have looked something up*
#: want to be resident, while reference material (timing analysis, widget
#: conventions, subsystem internals) is safe to leave lazy, because an agent
#: about to edit a widget knows to consult the widget section.
#:
#: WHAT THIS MODULE CAN ACTUALLY DELIVER IS NARROWER, and the difference has
#: been measured rather than assumed. The head is necessarily the file's first
#: N bytes: this module renders whatever the operator wrote, in the order they
#: wrote it, and MUST NOT reorder or edit their file to suit its own budget.
#: So the head holds the unconditional rules only where the file happens to
#: front-load them. On this repository's own AGENTS.md it largely does not —
#: measured at 8KiB, the head carries the test/lint/e2e gates and the xdist
#: worker-count rationale, while "never symlink a venv", "read the committed
#: ref", the version-bump rule, the merge tiers and the release-owner protocol
#: are all INDEX-ONLY. That is a property of the file's ordering, not a defect
#: this module may fix; front-loading is the file owner's call.
#:
#: The index is what makes that acceptable rather than fatal: those sections
#: carry imperative headings ("Never symlink one", "Read the committed ref,
#: not the working tree") which state the rule's gist in the row itself, and
#: they are one addressed read away. Note also that on this file they are past
#: the old 64KiB byte cut anyway, so they move from absent-and-invisible to
#: named-and-reachable.
#:
#: 8KiB was chosen over 6KiB and 12KiB: it carries the gates that apply to
#: every task regardless of subject while cutting ~19.7k billed tokens from a
#: fresh session. Raising it buys progressively less, since material further
#: in is increasingly subject-specific — exactly what an index serves well.
GUIDANCE_HEAD_BYTES = 8 * 1024

#: Deepest heading level the index lists. H3 is a real tuning knob rather than
#: an arbitrary depth: on this repository's own AGENTS.md it is the difference
#: between 11 rows and 33, and the H3s are where the individually actionable
#: rules live ("Never symlink one", "Read the committed ref"). Going deeper
#: costs index size for headings too fine-grained to be worth a separate read.
INDEX_MAX_HEADING_LEVEL = 3

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
    ``MAX_DIGEST_BYTES + 1`` probe determines truncation without ingesting the
    rest of an attacker-controlled file.
    """
    with _open_nofollow(path) as stream:
        probe = stream.read(MAX_DIGEST_BYTES + 1)
    return probe[:MAX_DIGEST_BYTES], len(probe) > MAX_DIGEST_BYTES


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


def _split_lines_like_read(data: bytes) -> list[str]:
    """Split exactly as the ``read`` tool does, because it is the consumer.

    THE INDEX'S LINE NUMBERS ARE A CONTRACT WITH ``read``. That tool decodes
    UTF-8 and calls ``str.splitlines()`` (``tools/builtin.py``
    ``_decode_text_lines``), then slices ``lines[start - 1 : end]``. Counting
    on ``\\n`` alone instead would disagree on SIX further separators that
    ``splitlines`` also breaks on -- ``\\v``, ``\\f``, ``\\x1c``-``\\x1e``,
    NEL, LS (U+2028) and PS (U+2029) -- and every occurrence before a heading
    shifts that heading's number by one, compounding down the file.

    That failure is silent and confident: the model lands on a plausible but
    WRONG range and follows the wrong rule, which is worse than finding
    nothing. A form feed or NEL pasted in from a table or a word processor is
    not exotic. So the split is not merely "similar" to read's, it is the
    same call on the same bytes; ``test_line_numbering_matches_the_read_tool``
    pins the two together so a change to either side breaks loudly.
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
    return text.splitlines()


def _scan_sections(
    path: Path, max_level: int = INDEX_MAX_HEADING_LEVEL
) -> tuple[list[_Section], int, bool, bool]:
    """Headings, total line count, whether the scan hit its ceiling, and
    whether a code fence was left open at EOF.

    Reads the file in bounded chunks and splits with :func:`_split_lines_like_read`
    so the emitted ranges resolve under ``read``. Only headings are retained,
    because the index must describe a file far larger than the prompt can
    hold. Fenced code blocks are tracked because ``#`` starts a comment in
    most of the shell snippets these files carry, and a comment indexed as a
    section sends a later ``read`` to the wrong range.
    """
    sections: list[_Section] = []
    total_lines = 0
    truncated_scan = False
    fence: str | None = None

    with _open_nofollow(path) as stream:
        raw_bytes = stream.read(MAX_SCAN_BYTES + 1)
    if len(raw_bytes) > MAX_SCAN_BYTES:
        truncated_scan = True
        raw_bytes = raw_bytes[:MAX_SCAN_BYTES]

    def collect(track_fences: bool) -> list[_Section]:
        """One pass. ``track_fences=False`` is the recovery pass below."""
        found: list[_Section] = []
        nonlocal fence
        fence = None
        for number, line in enumerate(lines, 1):
            stripped = line.lstrip()
            # ``` or ~~~ toggles; the closing fence must match the opener so a
            # ```python block containing ``` in prose does not end it early.
            if track_fences and (stripped.startswith("```") or stripped.startswith("~~~")):
                marker = stripped[:3]
                if fence is None:
                    fence = marker
                elif fence == marker:
                    fence = None
                continue
            if (track_fences and fence is not None) or not line.startswith("#"):
                continue
            level = len(line) - len(line.lstrip("#"))
            if level > max_level or not line[level:].startswith(" "):
                continue
            title = line[level:].strip()
            if not title:
                continue
            found.append(_Section(level, title, number))
        return found

    lines = _split_lines_like_read(raw_bytes)
    total_lines = len(lines)
    sections = collect(track_fences=True)
    unterminated_fence = fence is not None

    # A fence still open at EOF is far more often a formatting slip -- a
    # snippet whose closing ``` was forgotten -- than a genuine multi-KB code
    # block running to the end of a rules file. Believing it swallows every
    # heading after the slip and strands that whole tail unnamed, which is the
    # silent-loss failure this module exists to prevent. So when the file ends
    # mid-fence AND that cost us headings, re-scan ignoring fences: a spurious
    # row pointing at a shell comment is a far cheaper error than an
    # unreachable half of the operator's rules.
    if unterminated_fence:
        recovered = collect(track_fences=False)
        if len(recovered) > len(sections):
            sections = recovered

    # A section's span ends where the next same-or-higher heading begins.
    for index, section in enumerate(sections):
        end = total_lines
        for later in sections[index + 1 :]:
            if later.level <= section.level:
                end = later.start - 1
                break
        section.end = end
    return sections, total_lines, truncated_scan, unterminated_fence


def _open_nofollow(path: Path) -> BinaryIO:
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
                # Length is folded in because only the first MAX_DIGEST_BYTES
                # are hashed: without it, two oversized files sharing a 64KiB
                # prefix but differing in length dedup to one, and the survivor
                # would carry an index describing the wrong file.
                digest_state.update(str(found_here.stat().st_size).encode())
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
        return text, len(_split_lines_like_read(probe)), False
    head = probe[:GUIDANCE_HEAD_BYTES]
    # ``cut > 0``, not ``>= 0``: a file whose FIRST byte is a newline has its
    # only boundary at 0, and cutting there would empty the head entirely. In
    # that case the raw byte cut is the lesser evil, and the head is one
    # partial line -- which is why the count below is taken from the text that
    # actually ships rather than assumed to be whole lines.
    cut = head.rfind(b"\n")
    if cut > 0:
        head = head[:cut]
    text = head.decode("utf-8", errors="replace")
    # Counted with read's own splitter so head_lines and the index's numbering
    # come from one basis; see _split_lines_like_read.
    return text, len(_split_lines_like_read(head)), True


def _render_bare_pointer(shown: str, head_lines: int, total_lines: int | None) -> str:
    """Disclosure for an oversized file that yielded no listable sections.

    The floor this module must never fall below: the reader learns the file
    continues, where it is, and what to read. Without a section list the range
    is simply everything after the head.
    """
    if total_lines is not None and total_lines > head_lines:
        where = (
            f"`{shown}` ({total_lines} lines); the part not shown is "
            f"L{head_lines + 1}-{total_lines}"
        )
    else:
        where = f"`{shown}`; the part not shown begins at L{head_lines + 1}"
    return (
        f"\nThe rest of this file is NOT included above. It has no further "
        f"headings to index, so it cannot be listed by section. It is on disk "
        f"at {where}.\n\n"
        f"Before acting or answering on anything this file governs, you MUST "
        f"`read` that range \u2014 even when you believe you already know the "
        f"answer, and even when the part shown above seems to cover it. "
        f"Unlisted does not mean unimportant: this project's specific gates "
        f"and landmines may be stated only in the part you have not read."
    )


def _render_index(path: Path, shown: str, head_lines: int) -> str:
    """The section index that makes the non-resident remainder reachable.

    NEVER returns ``""`` for a file that has more content than the head. The
    caller appends whatever this returns, so an empty string there would ship
    a bare head with no path, no line count and no hint that the file
    continues -- content silently dropped AND unnamed, which is the exact
    failure this module exists to invert, and strictly worse than the byte cut
    it replaced (that at least said "truncated"). When there is nothing to
    list, this degrades to a bare pointer rather than to silence.
    """
    try:
        sections, total_lines, scan_truncated, unterminated = _scan_sections(path)
    except OSError:
        # The head still shipped, so say the file continues even though its
        # shape is unknown; total_lines is unavailable, hence the bare form.
        return _render_bare_pointer(shown, head_lines, None)
    # Level 1 is the document's title, not a section: it spans the whole file,
    # so an index row for it says nothing the path and line count do not. Its
    # span is still scanned, because an H2's range must end at the next H1 too.
    remaining = [s for s in sections if s.end > head_lines and s.level >= 2]
    if not remaining:
        # H1-only files, heading-free files, and files whose every heading sits
        # inside the head all land here. Nothing to index, but the tail is
        # still real and must stay reachable.
        return _render_bare_pointer(shown, head_lines, total_lines)
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
    # Belt and braces against the silent-loss class generally: if the listed
    # rows do not actually reach the end of the file, say so with the range
    # that does. A row set can be non-empty and still leave a tail unlisted
    # (e.g. every heading sits inside the head, or the scan stopped early), and
    # an unreachable tail is the one outcome this module must never produce.
    covered_to = max(section.end for section in remaining)
    if covered_to < total_lines:
        note += (
            f"\n(L{covered_to + 1}-{total_lines} is not under any listed "
            f"heading; read it directly)"
        )
    if unterminated:
        # Disclosed rather than silently absorbed: the reader should know the
        # index was built through a formatting slip, so a surprising-looking
        # row is explainable rather than mistaken for a real section.
        note += (
            "\n(this file ends inside an unclosed code fence; headings after "
            "it were indexed anyway so the tail stays reachable)"
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
