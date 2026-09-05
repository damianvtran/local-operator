"""Builtin tools for the new harness.

Why this module exists
----------------------
Tools declare as classes implementing ``AgentTool`` with per-tool
parameter schemas; the old local-operator instead injected Python callables
into executed code and rendered prose signatures into the prompt (the
``prompts.py`` reflection generator — the audit flagged that as the main
thing keeping the prompt at 176 KB). The rewrite adopts that shape: each
tool is an :class:`local_operator.harness.types.AgentTool` with a JSON Schema
derived from a pydantic parameter model, executed via native provider tool
calling.

Conventions every tool here follows:

- Parameter schema: a module-level pydantic model per tool;
  ``model_json_schema()`` output becomes ``AgentTool.parameters``. Field
  ``description`` strings are the model's only documentation the LLM sees.
- Parameter validation failures are returned as a clean
  ``invalid arguments:`` list — never a traceback — so the model can fix its
  call; truly unexpected exceptions are caught by the shared ``_guard``
  wrapper and returned as an error result carrying the traceback tail, so a
  buggy tool can never kill the turn.
- ``useless`` flags contextually worthless results (zero matches) so
  compaction may elide them once consumed; it is never combined with
  ``is_error``, and a useless result always carries
  ``details={"useless": True}`` so compaction can trust the payload.
- Approval tiers follow the read/write/exec model; the host's approval
  callback on ``ToolContext`` gates mutating side effects. Paths are always
  resolved before the prompt is built, so the user approves the exact file
  that will change; paths outside the workspace are flagged in the approval
  text and always require approval, even for read-tier tools.

The ``wake`` tool delegates its schedule maths to
``local_operator.harness.wake``, which is pure data plus a timer and costs
nothing to import; the tool itself is only advertised when the host actually
attached a scheduler to the ``ToolContext``.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import codecs
import contextlib
import difflib
import fnmatch
import json
import logging
import mimetypes
import os
import re
import signal as signal_module
import threading
import time
import traceback
import unicodedata
from collections import deque
from collections.abc import Awaitable, Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from rich.cells import cell_len

from local_operator.harness.approval import ask_approval
from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    AgentToolUpdate,
    ApprovalDescribeFn,
    AskQuestion,
    BrowserSurface,
    BrowserSurfaceProtocol,
    ImageContent,
    TextContent,
    ToolContext,
    ToolResult,
    VariableStoreProtocol,
    WakeSchedulerProtocol,
)
from local_operator.harness.wake import (
    WakeSchedule,
    build_wake_schedule,
    format_duration,
)
from local_operator.imaging import (
    IMAGE_INGEST_MAX_EDGE,
    IMAGE_JPEG_QUALITY,
    IMAGE_MAX_BYTES,
    IMAGE_MAX_PIXELS,
    bound_image_for_model,
)
from local_operator.media import ImageInfo, sniff_image_file
from local_operator.tools import group_reaper
from local_operator.tools.spill import (
    SPILL_ENTRY_LIMIT_BYTES,
    SPILL_SCHEME,
    SPILL_SEARCH_MATCH_LIMIT,
    SpillMeta,
    SpillRef,
    SpillStore,
    get_store,
    parse_handle,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared limits and helpers
# ---------------------------------------------------------------------------

#: Single combined budget for captured stdout+stderr (chars, not bytes, since
#: the output is one decoded transcript).
#:
#: MEASURED, not chosen by taste. Real log-like output on this repo encodes at
#: 3.80-4.05 chars/token under cl100k_base (full unit-suite ``pytest -v``:
#: 298,304 chars / 73,679 tokens = 4.05; ``grep -rn 'def '``: 3.95;
#: ``git log -p``: 3.80). At the previous 50 KiB cap ONE tool result cost
#: 12,538-13,100 tokens — over 40% of the 30,000-token start-context budget in
#: docs/REWRITE.md, spent on a single call. At 8 KiB the same three workloads
#: cost 2,086-2,169 tokens, a measured 6.1x reduction, and the clip still
#: carries ~55 lines of head and ~55 lines of tail, which is enough to hold a
#: pytest failure summary and the exit line — the two places the answer
#: actually lives. Anything the clip drops is recoverable by handle rather
#: than destroyed, which is what makes a cap this tight safe at all.
TOOL_OUTPUT_LIMIT_CHARS = 8 * 1024

#: Back-compat name for the bash-specific budget. Same value: there is one
#: per-result budget, and a second knob would let the two drift.
BASH_OUTPUT_LIMIT_CHARS = TOOL_OUTPUT_LIMIT_CHARS

#: Maximum number of matches returned by grep.
GREP_MATCH_LIMIT = 200
#: Maximum number of paths returned by glob.
GLOB_RESULT_LIMIT = 500
#: Default timeout for bash commands (seconds).
BASH_DEFAULT_TIMEOUT_SECONDS = 120.0
#: Hard cap on the per-command timeout; longer runs are a session bug, not a
#: tool feature.
BASH_MAX_TIMEOUT_SECONDS = 3600.0
#: Number of trailing traceback characters kept in an error result.
TRACEBACK_TAIL_CHARS = 2000

#: Files larger than this are refused by read as TEXT (serve 2MB+ blobs
#: through bash with head/tail instead); the cap serves the per-tool output
#: budget. Images are governed by :data:`READ_IMAGE_LIMIT_BYTES` instead.
READ_FILE_LIMIT_BYTES = 2 * 1024 * 1024
#: Byte ceiling for a file read as an IMAGE, 8x the text ceiling. The text cap
#: exists because bytes become context; an image's context cost is its PIXELS
#: (~w*h/750 tokens on Anthropic and OpenAI alike) and is bounded downstream by
#: :data:`READ_IMAGE_MAX_EDGE` no matter how large the source file is. Applying
#: the text cap to images refused ordinary inputs for no benefit and offered
#: nonsense advice while doing it: two of seven real PNG screenshots sampled on
#: this machine were 2.0-2.1 MB, and "use bash (head/tail)" is not a way to
#: look at a screenshot. What this cap actually limits is DECODE cost, so it is
#: paired with :data:`READ_IMAGE_MAX_PIXELS` — compressed bytes bound neither
#: the pixel count nor the RAM to hold it.
READ_IMAGE_LIMIT_BYTES = 16 * 1024 * 1024
#: Pixel, edge and byte bounds for an image block live in
#: :mod:`local_operator.imaging` — ``read`` is one of two callers that turn
#: bytes into an ``ImageContent`` (the composer's paste handler is the other),
#: and both have to bound identically or the unbounded one wedges the session
#: for both. Re-exported under the historical names because they are part of
#: this module's tested surface.
READ_IMAGE_MAX_PIXELS = IMAGE_MAX_PIXELS
READ_IMAGE_MAX_EDGE = IMAGE_INGEST_MAX_EDGE
READ_IMAGE_MAX_BYTES = IMAGE_MAX_BYTES
READ_IMAGE_JPEG_QUALITY = IMAGE_JPEG_QUALITY
#: Maximum lines read renders; larger files show the head plus a footer
#: telling the model to continue with a line range.
READ_LINE_CAP = 2000
#: Char budget for one ``read`` result. The line cap alone is not a budget:
#: 2,000 lines of ordinary source is ~80 KB (~20k tokens), so a single read of
#: a long file blew the same hole in the context that bash did. A file is its
#: own store — the footer names a ``range`` on the SAME path rather than a
#: spill handle, because spilling a copy of a file that is already on disk
#: would double the bytes for nothing.
READ_OUTPUT_LIMIT_CHARS = TOOL_OUTPUT_LIMIT_CHARS
#: Lines a footer suggests per expansion call. Sized so one page of ordinary
#: log text lands inside :data:`TOOL_OUTPUT_LIMIT_CHARS` (~55 head + ~55 tail
#: lines measured at 8 KiB, so 200 lines of typical 40-char output is the
#: right order and the page cap catches any overshoot). The point is that the
#: printed call SUCCEEDS: a suggestion whose own answer gets truncated teaches
#: the model that expansion does not work.
SPILL_PAGE_LINES = 200
#: Per-file size cap for grep; bigger files are skipped and counted.
GREP_FILE_LIMIT_BYTES = 1 * 1024 * 1024

#: Directory names never worth walking during grep (VCS internals, vendored
#: trees, build output). Dotdirs are pruned wholesale in addition.
_GREP_PRUNE_DIRS = frozenset({"__pycache__", "node_modules", "dist", "build", ".git", ".venv"})
#: Marker prefix on approval descriptions for targets outside the workspace.
OUTSIDE_WORKSPACE_MARKER = "[outside workspace]"
#: The OTHER reason a target escalates: it could not be resolved at all, so
#: nothing can be said about where it is. Distinct from the marker above because
#: they are different sentences — a path visibly under the workspace root
#: described as "outside the workspace" argues with itself, and a clause the user
#: can see is wrong is a clause they learn to ignore on the genuine escape.
UNRESOLVABLE_MARKER = "[unresolvable]"
#: Opens the row for a URL that could not be parsed. The sentence has to differ
#: from `browse:`, which asserts a destination — the whole point is that no
#: destination could be determined.
UNPARSED_URL_PREFIX = "unparsed url:"

#: Environment overrides that make common CLIs non-interactive.
NON_INTERACTIVE_ENV: dict[str, str] = {
    # Disable pagers so commands don't block on interactive views.
    "PAGER": "cat",
    "GIT_PAGER": "cat",
    "MANPAGER": "cat",
    "SYSTEMD_PAGER": "cat",
    "BAT_PAGER": "cat",
    "DELTA_PAGER": "cat",
    "GH_PAGER": "cat",
    "GLAB_PAGER": "cat",
    "PSQL_PAGER": "cat",
    "MYSQL_PAGER": "cat",
    "AWS_PAGER": "",
    "HOMEBREW_PAGER": "cat",
    "LESS": "FRX",
    # Disable terminal features that can block the process.
    "TERM": "dumb",
    "NO_COLOR": "1",
    "PYTHONUNBUFFERED": "1",
    # Disable editor and terminal credential prompts.
    "GIT_EDITOR": "true",
    "VISUAL": "true",
    "EDITOR": "true",
    "GIT_TERMINAL_PROMPT": "0",
    "SSH_ASKPASS": "/usr/bin/false",
    "CI": "1",
    # Package manager defaults for unattended execution.
    "npm_config_yes": "true",
    "npm_config_update_notifier": "false",
    "npm_config_fund": "false",
    "npm_config_audit": "false",
    "npm_config_progress": "false",
    "PNPM_DISABLE_SELF_UPDATE_CHECK": "true",
    "PNPM_UPDATE_NOTIFIER": "false",
    "YARN_ENABLE_TELEMETRY": "0",
    "YARN_ENABLE_PROGRESS_BARS": "0",
    # Cross-language/tooling non-interactive defaults.
    "CARGO_TERM_PROGRESS_WHEN": "never",
    "DEBIAN_FRONTEND": "noninteractive",
    "PIP_NO_INPUT": "1",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "TF_INPUT": "0",
    "TF_IN_AUTOMATION": "1",
    "GH_PROMPT_DISABLED": "1",
    "COMPOSER_NO_INTERACTION": "1",
    "CLOUDSDK_CORE_DISABLE_PROMPTS": "1",
}


#: Marker written where the middle of an output was removed. Kept as a public
#: name because tests and the browser paths reference it; the text now names
#: the recovery route instead of just announcing a loss.
BASH_TRUNCATION_MARKER = "\n\n... [output truncated] ...\n\n"


def _clip_head_tail(text: str, limit: int) -> tuple[str, str]:
    """``(head, tail)`` slices of ``text`` totalling at most ``limit`` chars.

    Both cuts snap INWARD to a line boundary, so neither end shows half a
    line. Half a line is not a cosmetic problem: a truncated ``File "x.py",
    line 12`` reads as a different path, and a model that acts on it edits the
    wrong file. When snapping would empty a side — one enormous line with no
    newline to snap to — the raw character slice is kept, because a fragment
    of the answer still beats none of it.
    """
    head_budget = limit // 2
    tail_budget = limit - head_budget

    head = text[:head_budget]
    cut = head.rfind("\n")
    if cut > 0:
        head = head[: cut + 1]

    tail = text[len(text) - tail_budget :]
    cut = tail.find("\n")
    if 0 <= cut < len(tail) - 1:
        tail = tail[cut + 1 :]

    return head, tail


def truncate_output(text: str, limit: int = TOOL_OUTPUT_LIMIT_CHARS) -> str:
    """Keep the head and tail of ``text`` when it exceeds ``limit``.

    The plain, store-free form: used where there is nothing to spill to (the
    browser's page text is re-fetchable by re-reading the page) and as the
    degraded path when a spill write fails. Prefer :func:`spill_truncate`,
    which produces the same shape but leaves the elided content recoverable.
    """
    if len(text) <= limit:
        return text
    head, tail = _clip_head_tail(text, limit - len(BASH_TRUNCATION_MARKER))
    return head + BASH_TRUNCATION_MARKER + tail


def _elision_span(text: str, head: str, tail: str) -> tuple[int, int, int]:
    """``(total_lines, first_elided_line, last_elided_line)``, all 1-based.

    Line numbers are what makes an expansion targeted rather than a blind
    page: the footer can say "lines 58-3970 are elided" and the model can ask
    for 40 of them. Counting is done on the same ``splitlines`` basis the
    store uses to serve a range, so the two agree by construction.
    """
    total = len(text.splitlines())
    head_lines = len(head.splitlines())
    tail_lines = len(tail.splitlines())
    return total, head_lines + 1, total - tail_lines


def _spill_footer(meta: SpillMeta, suggested: tuple[int, int] | None = None) -> str:
    """The recovery instructions that replace destroyed content.

    ONE footer per tool result, appended at the very end, never one per
    stream: a command that truncates both stdout and stderr has still produced
    a single output with a single handle, and repeating the instructions
    per-section spends the budget we just fought for on boilerplate the model
    reads twice.

    Spells out the EXACT call rather than describing it. A footer that says
    "the full output was saved" and stops is worse than no footer: the model
    knows something exists, cannot address it, and re-runs the command —
    paying the original cost again plus the truncation. Both usable forms are
    shown because they answer different questions: a range when the model
    knows where to look, a search when it does not.
    """
    handle = meta.handle
    first, last = suggested if suggested else (1, meta.lines)
    # Suggest ONE PAGE, not the whole gap. A footer that prints
    # range="462-3596" invites a call whose own answer is truncated at the
    # same budget, so the agent's first obedient follow-up lands it right back
    # where it started and it learns the handle does not work. Caught by
    # test_footer_names_a_call_that_actually_resolves, which runs the printed
    # call verbatim and requires the content back.
    page_end = min(last, first + SPILL_PAGE_LINES - 1)
    span = f"{first}-{page_end}"
    more = f" (of {first}-{last} elided; page through or search)" if page_end < last else ""
    partial = (
        ""
        if meta.complete
        else (
            f"\n  NOTE: output exceeded the {SPILL_ENTRY_LIMIT_BYTES // (1024 * 1024)} MB "
            "per-entry store cap; the stored copy is itself head+tail of the original."
        )
    )
    return (
        f"\n[Full output ({meta.lines} lines) is SAVED at {handle} — expand it, "
        f"do not re-run the command:\n"
        f'  read(path="{handle}", range="{span}")  -> those lines'
        f"{more}\n"
        f'  read(path="{handle}?q=<regex>")  -> find matching line numbers first, '
        f"then read a range around them{partial}]"
    )


def _spill(text: str, tool_name: str, context: ToolContext | None) -> SpillMeta | None:
    """Store ``text`` for later expansion; ``None`` when it could not be kept.

    Never raises. A store that cannot be written degrades the result to plain
    truncation, which is exactly the behaviour that shipped before this module
    existed — losing expansion is an inconvenience, failing the tool call is a
    bug.
    """
    return get_store().write(
        text,
        tool_name=tool_name,
        session_id=(context.session_id if context else "") or "",
    )


def _elide_inline(text: str, limit: int, offset: int = 0) -> tuple[str, tuple[int, int] | None]:
    """``(head + marker + tail, elided_span)`` with the span named IN the marker.

    The span is stated where the gap is, rather than only in the trailing
    footer, so a model scanning a two-stream result can see which lines are
    missing from WHICH stream. ``offset`` shifts the numbers into the
    coordinate space of the spilled copy, whose framing may differ from this
    fragment's (bash stores both streams under their banners in one entry).

    Returns a ``None`` span when nothing was elided.
    """
    if len(text) <= limit:
        return text, None
    # Two passes: build the marker from a first-pass clip to learn its true
    # length, then re-clip against the real budget. Sizing the clip with
    # ``len(BASH_TRUNCATION_MARKER)`` alone would overshoot the limit by the
    # length of the span annotation, and the limit is the whole point.
    head, tail = _clip_head_tail(text, limit - len(BASH_TRUNCATION_MARKER))
    total, first, last = _elision_span(text, head, tail)
    marker = _elision_marker(last - first + 1, total, first + offset, last + offset)
    head, tail = _clip_head_tail(text, limit - len(marker))
    total, first, last = _elision_span(text, head, tail)
    span = (first + offset, last + offset)
    marker = _elision_marker(last - first + 1, total, span[0], span[1])
    return head + marker + tail, span


def _elision_marker(elided: int, total: int, first: int, last: int) -> str:
    """The in-band gap marker.

    Keeps :data:`BASH_TRUNCATION_MARKER` as an exact substring — renderers and
    tests key off that literal — and adds the line span on its own line, so
    the model can see WHICH lines are missing right where they are missing
    rather than only in the trailing footer.
    """
    return (
        f"{BASH_TRUNCATION_MARKER.rstrip()}\n"
        f"[{elided} of {total} lines elided — they are lines {first}-{last} "
        f"of the saved output]\n\n"
    )


def spill_truncate(
    text: str,
    tool_name: str,
    context: ToolContext | None,
    limit: int = TOOL_OUTPUT_LIMIT_CHARS,
) -> tuple[str, dict[str, Any] | None]:
    """``(display_text, spill_details)`` for one oversized output.

    ``spill_details`` is ``{"spill": {...}}`` ready to merge into a
    ``ToolResult.details`` mapping, or ``None`` when nothing was spilled
    (output fit, or the store refused it). ``details`` never reaches a
    provider, so recording the handle there costs no prompt tokens while still
    letting renderers, transcripts and compaction see what happened.
    """
    if len(text) <= limit:
        return text, None
    meta = _spill(text, tool_name, context)
    if meta is None:
        return truncate_output(text, limit), None
    body, span = _elide_inline(text, limit)
    return body + _spill_footer(meta, span), {"spill": _spill_detail(meta)}


def _spill_detail(meta: SpillMeta) -> dict[str, Any]:
    """The ``details['spill']`` payload. One shape, so the transcript writer
    and the renderers cannot disagree about the key names."""
    return {
        "handle": meta.handle,
        "lines": meta.lines,
        "bytes": meta.bytes,
        "complete": meta.complete,
    }


def _safe_cwd(context: ToolContext | None) -> str:
    return context.cwd if context and context.cwd else "."


def _resolve_workspace_path(raw: str, cwd: str) -> tuple[Path, bool, bool]:
    """Resolve a tool-supplied path to an absolute ``Path``.

    ``~`` is expanded, relative paths join onto ``cwd``, and the result is
    fully resolved. Returns ``(path, inside, resolvable)``. ``inside`` is True
    only when the resolved path is known to stay within the resolved workspace
    root — approval prompts always show the resolved path, and a target that is
    not ``inside`` escalates approval even for read-tier tools.

    ``resolvable`` separates the two ways ``inside`` can be False, because they
    are different sentences to the user: a path that resolved and lies elsewhere,
    versus one that could not be resolved at all. Both escalate identically.
    """
    # `expanduser()` raises RuntimeError when `~user` names nobody the platform
    # can resolve — the second of the two sites the describer fix named, and the
    # one on the path every write/exec approval runs through. Falling back to the
    # unexpanded string keeps the sentence buildable: an unresolvable `~` is then
    # treated as a relative path under the workspace, which is where a resolved
    # target that escapes gets caught anyway.
    try:
        root = Path(cwd).expanduser().resolve()
    except RuntimeError:
        root = Path(cwd).resolve()
    try:
        candidate = Path(raw).expanduser()
    except RuntimeError:
        candidate = Path(raw)
    path = candidate if candidate.is_absolute() else root / candidate
    try:
        path = path.resolve()
    except (OSError, ValueError):
        # `resolve()` stats the path, so it raises on more than a missing parent:
        # an embedded NUL is a `ValueError` from the lstat itself, and a symlink
        # loop or a permission wall is an `OSError`. Unhandled, a model-supplied
        # `a\x00b` took down the approval prompt it was being asked about — the
        # same shape as the `expanduser` crash above, one line further on.
        #
        # Reported OUTSIDE, not inside. A path that cannot be resolved cannot be
        # shown to be within the workspace, and this verdict decides whether the
        # user is warned; the honest answer when the check cannot be made is the
        # one that still asks. The tool's own open() will fail afterwards anyway.
        return path, False, False
    try:
        path.relative_to(root)
    except ValueError:
        return path, False, True
    return path, True, True


#: Two or more consecutive spaces — the part of a name a flattened prompt line
#: cannot show literally.
_SPACE_RUN = re.compile(r"  +")


def _display_target(text: str) -> str:
    """A target rendered so that what the user reads IS what the tool will use.

    Plain text passes through. Anything whose displayed form would differ from
    the real string — leading or trailing whitespace, an embedded newline, a
    control byte — is quoted with Python escapes instead, because the prompt is
    sanitised into one inert line downstream and a silent clean-up is the same
    defect as naming the wrong file: `notes.md ` and `notes.md` are two files,
    and `a\x00b` is not `ab`.
    """
    # The test is against what SURVIVES the prompt sanitiser, not just against
    # leading and trailing space: it collapses internal runs too, so `/ws/  a.png`
    # would reach the user as `/ws/ a.png` — a different file, named silently.
    if text and text.isprintable() and text == " ".join(text.split()):
        return text
    # Quoting alone is not enough: the prompt sanitiser collapses whitespace runs
    # AFTER this, so a quoted `'/ws/  a.png'` still reached the user as
    # `'/ws/ a.png'`. Each space in a run becomes an explicit escape, which no
    # later flattening can touch and which a technical reader can count.
    return _SPACE_RUN.sub(lambda match: "\\x20" * len(match.group()), repr(text))


def _approval_description(path: Path, inside: bool, action: str, resolvable: bool = True) -> str:
    """Approval prompt text for ``action`` on a RESOLVED path (the user must
    approve the exact target, not the raw string the model typed).

    Two markers, one escalation. Both mean "this needs a closer look", and the
    renderer treats them identically at every width where the clause collapses to
    `!`; they differ only in the sentence spelled out when there is room for one.
    """
    if inside:
        marker = ""
    elif resolvable:
        marker = f"{OUTSIDE_WORKSPACE_MARKER} "
    else:
        marker = f"{UNRESOLVABLE_MARKER} "
    return f"{marker}{action}: {_display_target(str(path))}"


#: Cell budget for the free-text body an approval sentence quotes. The prompt is
#: read at a glance and shares its row with the host's ``Allow <tool>?`` prefix,
#: so the body is bounded rather than wrapped.
APPROVAL_BODY_CELLS = 60


def _truncate_approval_body(text: str, width: int = APPROVAL_BODY_CELLS) -> str:
    """Bound ``text`` to ``width`` CELLS, ending in the app's ellipsis.

    Measured in cells, not characters: a CJK body clipped by ``len()`` rendered
    138 cells against an intended 60 and wrapped the approval prompt onto a
    second line, because every character in it is two cells wide. The ellipsis
    is ``…`` for the same reason the rest of the TUI uses it — an ASCII ``...``
    in one truncation and ``…`` in another renders two styles in one frame.

    Deliberately a local reimplementation rather than an import of
    ``tui.widgets.tool_card.truncate_cells``: this module runs headless (the
    stdin approval gate has no TUI at all) and must not pull a Textual widget
    module in for a string bound.
    """
    if width <= 0 or not text:
        return ""
    if cell_len(text) <= width:
        return text
    ellipsis = "…"
    target = width - cell_len(ellipsis)
    out: list[str] = []
    used = 0
    for char in text:
        size = cell_len(char)
        if used + size > target:
            break
        out.append(char)
        used += size
    return "".join(out).rstrip() + ellipsis


def _describe_shell_approval(args: dict[str, Any], cwd: str) -> str:
    """``run: <command>`` — the command IS the decision for an exec-tier call.

    Kept in the argument's own order and NOT reformatted: a user authorising a
    shell command needs the text that will actually run, and re-quoting it would
    make the prompt and the executed string differ.
    """
    command = str(args.get("command") or "").strip()
    # Quoted when it spans lines: the sanitiser downstream collapses newlines to
    # spaces so a command cannot forge a second prompt, and a silently joined
    # two-line command would read as one command that was never typed.
    return f"run: {_display_target(command)}" if command else ""


def _describe_path_approval(action: str, key: str = "path") -> ApprovalDescribeFn:
    """``<action>: <resolved path>``, marked when the path leaves the workspace.

    Resolved rather than raw, and marked from the SAME resolver the tool itself
    uses, so the sentence the user answers names the file the tool will touch —
    `../../etc/hosts` and `~/x` are the two forms where the raw string and the
    target genuinely differ.
    """

    def describe(args: dict[str, Any], cwd: str) -> str:
        # NOT stripped: `execute_write` and `execute_edit` pass the raw string to
        # the resolver, and " notes.md" and "notes.md" are different files on a
        # POSIX filesystem. A prompt that quietly normalises names a file the tool
        # will not touch.
        raw = str(args.get(key) or "")
        if not raw.strip():
            return ""
        try:
            path, inside, resolvable = _resolve_workspace_path(raw, cwd or ".")
        except (OSError, ValueError, RuntimeError):
            # An unresolvable path is still worth naming; the tool will fail with
            # its own error, and a prompt that says nothing is worse than one
            # that quotes what the model asked for.
            return f"{action}: {_display_target(raw)}"
        return _approval_description(path, inside, action, resolvable)

    return describe


def _describe_wake_approval(args: dict[str, Any], cwd: str) -> str:
    """``schedule: <when> — <message>`` (or the operation for list/cancel).

    Wake is the one tool that arms an UNATTENDED future turn, so the decision is
    when it will run, how often, and what it will be told — not the parameter
    shape. The recurrence is part of the sentence because "once in 30m" and
    "every 30m forever" are different commitments and the difference is one
    word.

    Keys come from :class:`WakeParams`, which is ``extra="forbid"``: ``op``,
    ``message``, ``in`` (aliased, so the raw key is what arrives here), ``at``,
    ``every``, ``until``, ``limit``, ``id``. The first version of this function
    invented `action`/`when`/`prompt`, which cannot appear — so it silently never
    ran and the most dangerous tool in the set kept showing a JSON dump.
    """
    op = str(args.get("op") or "create").strip()
    if op == "cancel":
        identifier = str(args.get("id") or "").strip()
        return f"cancel wake: {identifier}" if identifier else "cancel wake"
    if op != "create":
        return f"wake: {op}"

    first = str(args.get("in") or args.get("at") or "").strip()
    every = str(args.get("every") or "").strip()
    # Plain ASCII words. The glyph this used to carry could not be gated: the
    # check was `cell_len("⟳") == 1`, which is a static Unicode width table and
    # not a terminal capability probe, so it measured 1 on every host and the
    # fallback was unreachable. This sentence also reaches the headless stdin
    # gate, where none of the TUI's glyph machinery runs at all, so it earns
    # nothing by being clever — and the BOUND already leads, which is what makes
    # two commitments differ at their first token.
    if first and every:
        when = f"{first} then every {every}"
    elif every:
        when = f"every {every}"
    else:
        when = first

    bound = ""
    if every:
        until = str(args.get("until") or "").strip()
        limit = args.get("limit")
        if until:
            bound = f" until {until}"
        elif isinstance(limit, int):
            bound = f" {limit}x"
        else:
            # The SAME slot and the same shape as a count, because an unbounded
            # recurrence is the one wake that never stops on its own and it must
            # not be the only bound rendered in a different grammar — the shape
            # that most needs emphasis was the one wearing parentheses.
            bound = " forever"

    message = " ".join(str(args.get("message") or "").split())
    # The BOUND leads the interval. Trailing, it was the last token on the row
    # and therefore the first thing a head-keeping truncation cut, so a wake
    # firing eight times and one that never stops painted the same text at three
    # widths — the collision this sentence was restructured to remove, recreated
    # inside the slot that removed it.
    head = f"schedule:{bound} {when}" if when else "schedule"
    return f"{head} — {message}" if message else head


def _describe_task_approval(args: dict[str, Any], cwd: str) -> str:
    """``subagent: <label>`` — the label names the child being started.

    Only the label can practically be shown: spawning a subagent runs a fresh
    child session whose whole prompt is the value of ``prompt``, and dumping
    that into the approval row would make the decision to start the child
    hinge on prose the user is not going to read from a gate line. The label
    is a short name the caller chose precisely to stand in for the work.
    """
    label = " ".join(str(args.get("label") or "").split())
    return f"subagent: {label}" if label else "subagent"


#: Browser actions whose whole effect is "go somewhere". Everything else acts on
#: the page that is already open, and says so.
NAVIGATING_BROWSER_ACTIONS = frozenset({"open", "goto", "navigate"})


def _describe_browser_approval(args: dict[str, Any], cwd: str) -> str:
    """``browse: <host/path>`` / ``browser: <action>`` — the site, then the verb.

    ``https://`` is dropped for the same reason ``$HOME`` collapses to ``~``: it
    is on every URL, it decides nothing, and it was eating eight of the forty
    cells a narrow prompt has — at 32 columns the row read `browse: http…` and
    named no site at all. A NON-https scheme is kept, because "this fetch is not
    encrypted" is exactly the kind of thing this prompt exists to surface.
    """
    raw_url = str(args.get("url") or "").strip()
    shown = _display_url(raw_url)
    # `None` means `_display_url` could not build a destination from this string,
    # so the row must not open with `browse:` as though it had. Reported rather
    # than inferred: deciding it by comparing against a re-derived opaque form
    # made the branch depend on whether the quoting happened to differ, and a
    # test could not tell the two paths apart.
    url_unparsed = shown is None
    url = _opaque_url(raw_url) if url_unparsed else shown
    # LOWERCASED, because `execute_browser` lowercases before it dispatches and
    # `BrowserParams.action` is a bare `str` with no enum. Comparing the raw value
    # meant `SCREENSHOT` fell past the screenshot branch: the prompt said
    # `browser: SCREENSHOT` — no path, no outside-workspace marker — while the
    # call wrote a PNG to whatever absolute path it was given. The action string
    # is model-controlled, so the case is model-controlled too.
    action = str(args.get("action") or "").strip().lower()
    # A `url` argument is only what the call DOES when the action navigates.
    # `click` and `type` carry the page they act on for context, and announcing
    # "browse: <url>" for them describes a navigation that will not happen while
    # hiding the interaction that will.
    if action == "screenshot":
        # The one browser action whose effect is on the FILESYSTEM. It rides the
        # write gate because it writes, so the prompt names what it writes —
        # resolved through the tool's own resolver and marked when it leaves the
        # workspace, exactly like `write`. Without this the row said
        # `browser: screenshot` while the call landed a PNG on /etc.
        # NOT stripped, for the same reason `_describe_path_approval` is not:
        # `_browser_screenshot` resolves the raw string, so `"  shot.png"` is a
        # different file from `"shot.png"` and the prompt must name the one that
        # will actually be written. Only the emptiness test is stripped.
        raw_path = str(args.get("path") or "")
        if not raw_path.strip():
            return "screenshot to a temporary file"
        try:
            path, inside, resolvable = _resolve_workspace_path(raw_path, cwd or ".")
        except (OSError, ValueError):
            return f"screenshot: {_display_target(raw_path)}"
        return _approval_description(path, inside, "screenshot", resolvable)
    if url and action in NAVIGATING_BROWSER_ACTIONS:
        if url_unparsed:
            return f"{UNRESOLVABLE_MARKER} {UNPARSED_URL_PREFIX} {url}"
        return f"browse: {url}"
    if action and url:
        if url_unparsed:
            # ONE label, not two. `{action}: unparsed url: <raw>` spent two labels
            # before naming anything, so the row protected the words `unparsed
            # url` as though they were the target: 40 of 71 widths painted no
            # character of the URL at all, and below 40 the ACTION was gone while
            # the label survived. The marker already says the URL could not be
            # read, so the action can keep its own slot and the string gets the
            # rest.
            return f"{UNRESOLVABLE_MARKER} {action}: {url}"
        return f"{action}: {url}"
    return f"browser: {action}" if action else ""


def _opaque_url(raw: str) -> str:
    """A URL this module could not parse, presented as DATA, not a destination.

    Paired with :data:`UNPARSED_URL_PREFIX` by the caller, so the row names the
    string without claiming to know where it points. The prefix is what does the
    work: `_display_target` quotes only a space-run or a non-printable, so the
    payload this exists for — a hostile URL of ordinary printable characters —
    comes back bare, and it is the label in front of it, not quotation, that
    stops the row reading as a destination.

    `_display_url`'s whole contract is "host first, no userinfo", and the two
    exits that could not honour it used to return the caller's string verbatim —
    so a row that normally reads `browse: evil.test/x` instead led with whatever
    the attacker put in front of the `@`. Quoting it makes the row read as the
    raw text the model supplied rather than as a claim about where the fetch
    goes, which is the same distinction the write describer draws for a path it
    cannot resolve.
    """
    return _display_target(raw)


def _punycode_host(host: str) -> str:
    """A non-ASCII host shown the way a browser shows it: punycode.

    `аpple.com` with a Cyrillic `а` is pixel-identical to `apple.com` in most
    terminal fonts and resolves to `xn--pple-43d.com`. Every browser displays
    the encoded form for exactly this reason, and a security prompt has a
    stronger obligation than a browser does: it is asking someone to authorise
    the fetch. The `hostname` attribute lowercases but does not encode.
    """
    if host.isascii():
        return host
    try:
        return host.encode("idna").decode("ascii")
    except UnicodeError:
        # A host IDNA cannot encode is not a host anyone should be waved past.
        return host.encode("unicode_escape").decode("ascii")


def _display_url(raw: str) -> str | None:
    """The URL as a DESTINATION: host first, no userinfo, no redundant scheme.

    `https://` is dropped because it is on every URL and discriminates nothing —
    the same trade `~` makes for `$HOME`. A non-https scheme is KEPT: "this fetch
    is not encrypted" is exactly what this prompt exists to surface.

    Userinfo is dropped entirely. `http://accounts.google.com@evil.test/x` is a
    request to **evil.test**, and it is the one part of a URL an attacker fully
    controls AND the one part a left-anchored row never truncates — so the
    never-cut opening of the sentence would have been attacker-chosen text that
    is not where the browser goes.
    """
    if not raw:
        return raw
    try:
        parts = urlsplit(raw)
    except ValueError:
        # Unparseable, so this function cannot say where the fetch goes — and
        # echoing the caller's string is the one thing it must not do. U+FF20
        # FULLWIDTH COMMERCIAL AT reaches here: `_validate_browser_url` gates only
        # on the scheme prefix, and CPython's `_checknetloc` refuses the netloc
        # under NFKC normalisation, so `http://accounts.google.com＠evil.test/x`
        # fell out verbatim and the row led with `accounts.google.com`. Whether a
        # WebView normalises that to `@` is unproven, but a prompt that cannot
        # parse a URL has no business asserting a destination from it.
        return None
    if not parts.hostname:
        # Same contract, different cause: with no host there is no destination to
        # put first, so the sentence this function exists to build cannot be made.
        return None
    try:
        port = parts.port
    except ValueError:
        # `urlsplit` defers port validation to attribute access, so a URL like
        # `http://h:99999/` parses and then raises on `.port` — outside the try
        # that exists to keep a malformed URL from crashing the prompt.
        #
        # Dropped, NOT degraded to `raw`. Returning the raw string handed back the
        # exact input this function exists to sanitise, and one extra character
        # was enough to reach it: `http://accounts.google.com@evil.test:99999/x`
        # painted with its userinfo intact, so a left-anchored prompt at 46
        # columns affirmatively named `accounts.google…` while the browser went to
        # evil.test — and a homograph host kept its lookalike spelling. Worse than
        # having no describer at all, which at least reads as raw data rather than
        # as a confident destination. `parts.hostname` is already parsed and
        # guarded above, so the sentence can be built without the port.
        port = None
    host = _punycode_host(parts.hostname)
    if ":" in host:
        # An IPv6 literal needs its brackets back: `::1:8080` does not say where
        # the address ends and the port begins, on a row whose only job is to
        # state the destination unambiguously.
        host = f"[{host}]"
    if port:
        host = f"{host}:{port}"
    tail = parts.path or ""
    if parts.query:
        tail += f"?{parts.query}"
    # A non-https scheme LEADS. Trailing it read well and truncated first, so
    # from 47 columns down an http URL and an https one painted identically —
    # and the row spent four widths rendering the stub `(htt…`. Leading, it is
    # part of the head that head-keeping truncation preserves, and https (the
    # safe case, and the overwhelming majority) still costs nothing.
    lead = "" if parts.scheme == "https" else f"{parts.scheme}! "
    return f"{lead}{host}{tail}"


def _error(tool_call_id: str, tool_name: str, message: str) -> ToolResult:
    """Build a non-throwing error result (loop never raises into the model)."""
    return ToolResult(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        content=[TextContent(text=message)],
        is_error=True,
    )


def _text(
    tool_call_id: str,
    tool_name: str,
    text: str,
    *,
    useless: bool = False,
    details: dict[str, Any] | None = None,
) -> ToolResult:
    """Build a plain-text result; ``details`` carries structured payload for
    renderers and compaction pruning (e.g. ``path`` for file tools)."""
    return ToolResult(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        content=[TextContent(text=text)],
        details=details,
        useless=useless,
    )


def _image(
    tool_call_id: str,
    tool_name: str,
    caption: str,
    payload: bytes,
    mime_type: str,
    *,
    details: dict[str, Any] | None = None,
) -> ToolResult:
    """Build an image result: a one-line caption FOLLOWED BY the image block.

    The caption is not decoration. An image block arrives in the transcript
    with no filename, no format and no dimensions attached — a bare one leaves
    the model unable to say what it is looking at, whether the read it asked
    for is the thing it got, or even that the call succeeded rather than
    silently returning nothing. It leads for the same reason: every text-only
    consumer in the stack (``ToolResult.text``, compaction's truncation, the
    TUI transcript row) sees the caption and nothing else, so the caption is
    the entire result for all of them.
    """
    return ToolResult(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        content=[
            TextContent(text=caption),
            ImageContent(data=base64.b64encode(payload).decode("ascii"), mime_type=mime_type),
        ],
        details=details,
    )


def _validation_error(tool_call_id: str, tool_name: str, exc: ValidationError) -> ToolResult:
    """One ``invalid arguments:`` line per field — no traceback. The model can
    correct its call from the message; the stack trace could not."""
    lines = [
        f"- {'.'.join(str(part) for part in err['loc']) or '<root>'}: {err['msg']}"
        for err in exc.errors()
    ]
    return _error(tool_call_id, tool_name, "invalid arguments:\n" + "\n".join(lines))


#: The shape every ``execute_*`` in this module has. It differs from
#: ``ToolExecuteFn`` only in accepting ``context=None``, which the bare-tool
#: tests rely on; a function that accepts None also satisfies the stricter
#: harness signature, so these still slot into ``AgentTool.execute``.
ToolExecutor = Callable[
    [
        str,
        dict[str, Any],
        AbortSignal | None,
        Callable[[AgentToolUpdate], None] | None,
        ToolContext | None,
    ],
    Awaitable[ToolResult],
]

#: Process-wide path transaction locks shared by ``read``, ``edit`` and
#: ``write``. Tool ``concurrency="exclusive"`` is scoped to one AgentLoop;
#: parent/child sessions own separate loops and can otherwise enter two
#: thread-backed read-modify-write transactions on the same file at once.
#: Fixed stripes keep the table bounded (no lock per model-controlled path);
#: a collision only serializes two unrelated mutations, which is a safe,
#: rare cost. Reads take the same stripe so they cannot observe a truncated
#: file between a writer's open and close.
_FILE_TRANSACTION_LOCKS = tuple(threading.Lock() for _ in range(64))


def _file_path_identity(path: Path) -> str:
    """Coalesce spelling aliases before a new file has an inode to lock.

    macOS can resolve composed/decomposed Unicode names to the same file while
    ``Path.resolve`` preserves their different spellings. Both the scheduler
    and cross-session transaction locks must share this identity, otherwise
    two creators can enter concurrently before either can observe an inode.
    Folding/normalizing may conservatively serialize distinct files on other
    filesystems; it only affects coordination, never the actual I/O path.
    Normalize after folding because folding can introduce decomposed text.
    """
    return unicodedata.normalize("NFC", str(path.resolve(strict=False)).casefold())


def _file_resource_keys(args: dict[str, Any], cwd: str) -> tuple[str, ...]:
    """Declare mutation conflicts without weakening the transaction locks.

    Always include the canonical path (also for new files), plus device/inode
    for existing hardlinks. Resolution failures raise so the loop can retain
    the exclusive barrier. The loop runs this filesystem probe off-thread.
    """
    raw = args.get("path")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("mutation requires a path")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path(cwd) / path
    keys = ["file:path:" + _file_path_identity(path)]
    try:
        stat = path.stat()
    except FileNotFoundError:
        pass
    else:
        keys.append(f"file:inode:{stat.st_dev}:{stat.st_ino}")
    return tuple(keys)


@contextlib.contextmanager
def _file_transaction(path: Path) -> Iterator[None]:
    """Lock the canonical path and, when it exists, its filesystem identity.

    The path stripe is always held, so a transaction that creates a file
    cannot be bypassed by a second caller that observes the new inode. The
    inode stripe additionally coalesces hardlinks. Symlink/case/Unicode aliases
    share the canonical path key. Multiple stripes are acquired in
    numeric order to make overlapping alias sets deadlock-free.
    """
    canonical = ("path", _file_path_identity(path))
    keys: list[object] = [canonical]
    try:
        stat = path.stat()
    except OSError:
        pass
    else:
        keys.append(("inode", stat.st_dev, stat.st_ino))
    indices = sorted({hash(key) % len(_FILE_TRANSACTION_LOCKS) for key in keys})
    locks = [_FILE_TRANSACTION_LOCKS[index] for index in indices]
    for lock in locks:
        lock.acquire()
    try:
        yield
    finally:
        for lock in reversed(locks):
            lock.release()


def _guard(tool_name: str) -> Callable[[ToolExecutor], ToolExecutor]:
    """Wrap an execute coroutine so unexpected exceptions become error results.

    The harness contract is that tools never throw into the loop: provider
    error paths (Anthropic rejects empty is_error blocks) and retry logic all
    assume a ToolResult comes back. The traceback tail is included so the
    model can self-correct and we can debug from transcripts.
    """

    def decorator(fn: ToolExecutor) -> ToolExecutor:
        async def wrapper(
            tool_call_id: str,
            args: dict[str, Any],
            signal: AbortSignal | None = None,
            on_update: Callable[[AgentToolUpdate], None] | None = None,
            context: ToolContext | None = None,
        ) -> ToolResult:
            try:
                return await fn(tool_call_id, args, signal, on_update, context)
            except Exception:  # noqa: BLE001 — boundary: nothing may escape
                return _error(
                    tool_call_id,
                    tool_name,
                    f"Tool '{tool_name}' failed unexpectedly:\n"
                    f"{traceback.format_exc()[-TRACEBACK_TAIL_CHARS:]}",
                )

        wrapper.__name__ = f"execute_{tool_name}"
        wrapper.__qualname__ = wrapper.__name__
        return wrapper

    return decorator


async def _check_approval(context: ToolContext | None, tier: str, description: str) -> bool:
    """Ask the host for approval; True means proceed.

    No approval hook installed -> auto-approved (CLI --yolo and headless tests
    rely on this). A hook returning False denies the action without error
    state beyond a plain refusal message.

    Routed through ``ask_approval`` for the same reason the loop's tier gate
    is: the two paths must put the question to the host the SAME way, or a
    self-gating tool's ask arrives without the provenance a tier-gated one
    carries and a host scoping its answer sees half the picture.
    """
    request_approval = getattr(context, "request_approval", None) if context else None
    if request_approval is None:
        return True
    return await ask_approval(request_approval, tier, description, getattr(context, "job_id", None))


async def _run_with_abort(
    coro: Awaitable[Any],
    signal: AbortSignal | None,
    on_abort: Callable[[], None],
) -> tuple[Any, bool]:
    """Race ``coro`` against the abort signal.

    Returns ``(result_or_None, aborted)``. On abort ``on_abort`` runs (e.g.
    process kill) before the coroutine is cancelled, so resources held by
    the awaited call are released deterministically instead of being
    abandoned. A signal already aborted at entry STILL runs ``on_abort`` and
    closes the pending coroutine — the old early return skipped both, which
    leaked the spawned child and raised "coroutine was never awaited"
    (RT-01). Callers that must not spawn at all should check
    ``signal.aborted`` before creating the coroutine.
    """
    if signal is not None and signal.aborted:
        on_abort()
        if asyncio.iscoroutine(coro):
            coro.close()
        return None, True
    if signal is None:
        return await coro, False
    waiter = asyncio.create_task(signal.wait())
    work = asyncio.ensure_future(coro)
    done, _pending = await asyncio.wait({waiter, work}, return_when=asyncio.FIRST_COMPLETED)
    if work in done:
        waiter.cancel()
        with contextlib.suppress(BaseException):
            await waiter
        return work.result(), False
    on_abort()
    work.cancel()
    with contextlib.suppress(BaseException):
        await work
    return None, True


# ---------------------------------------------------------------------------
# bash
# ---------------------------------------------------------------------------


class BashParams(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    command: str = Field(description="Shell command to run (executed via /bin/sh -c).")
    timeout: float = Field(
        default=BASH_DEFAULT_TIMEOUT_SECONDS,
        gt=0,
        le=BASH_MAX_TIMEOUT_SECONDS,
        description="Max seconds before the command is killed.",
    )
    background: bool = Field(
        default=False,
        description=(
            "Start the command and return a job id immediately instead of "
            "waiting. Use for long work (builds, training, terraform apply, "
            "pipeline polling); follow it with jobs(op='peek') to read new "
            "output as it arrives, and jobs(op='cancel') to stop it. 'timeout' "
            "still bounds the run."
        ),
    )


class _BashOutput:
    """Bound retention while the pipe is drained, keeping both diagnostic ends.

    The spill store already caps a transcript at 4 MiB. Retaining arbitrarily
    many raw bytes until that write bought no recoverability and let a noisy
    process exhaust the host. Each pipe keeps at most that existing cap, so
    outputs that fit the store remain complete and larger ones keep head/tail.
    The explicit gap is part of the stored text; its line numbers describe the
    retained copy, never pretend to address discarded original lines.
    """

    def __init__(self, limit: int = SPILL_ENTRY_LIMIT_BYTES) -> None:
        self.limit = limit
        self.total_bytes = 0
        self.head = bytearray()
        self.tail: deque[bytes] = deque()
        self.tail_bytes = 0

    @property
    def retained_bytes(self) -> int:
        return len(self.head) + self.tail_bytes

    @property
    def omitted_bytes(self) -> int:
        return self.total_bytes - self.retained_bytes

    def append(self, chunk: bytes) -> None:
        self.total_bytes += len(chunk)
        head_room = max(self.limit // 2 - len(self.head), 0)
        self.head.extend(chunk[:head_room])
        chunk = chunk[head_room:]
        if chunk:
            self.tail.append(chunk)
            self.tail_bytes += len(chunk)
        ceiling = self.limit - len(self.head)
        while self.tail_bytes > ceiling:
            first = self.tail.popleft()
            excess = self.tail_bytes - ceiling
            if len(first) > excess:
                self.tail.appendleft(first[excess:])
                self.tail_bytes -= excess
            else:
                self.tail_bytes -= len(first)

    def chunks(self) -> list[bytes]:
        return [bytes(self.head), *self.tail]

    def decode(self) -> str:
        if not self.omitted_bytes:
            return b"".join(self.chunks()).decode("utf-8", errors="replace")
        return (
            self.head.decode("utf-8", errors="replace")
            + f"\n[retention limit: {self.omitted_bytes} bytes omitted]\n"
            + b"".join(self.tail).decode("utf-8", errors="replace")
        )


class _PipeRedactor:
    """Delay only a possible credential suffix before publishing pipe bytes.

    Redacting each read independently leaks a secret split across reads. Keep
    enough undecided text for the longest injected credential, and never cut
    through a complete match. UTF-8 decoding is incremental for the same
    reason. Retained output and live job tails receive the same safe bytes.
    """

    def __init__(self, credentials: dict[str, str]) -> None:
        self.secrets = sorted(
            {value for value in credentials.values() if value}, key=len, reverse=True
        )
        self.lookbehind = max((len(value) for value in self.secrets), default=1) - 1
        self.pending = ""
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def feed(self, chunk: bytes, *, final: bool = False) -> bytes:
        text = self.pending + self.decoder.decode(chunk, final=final)
        cut = len(text) if final else max(len(text) - self.lookbehind, 0)
        while True:
            previous_cut = cut
            for secret in self.secrets:
                start = text.find(secret, max(cut - len(secret) + 1, 0))
                if 0 <= start < cut < start + len(secret):
                    cut = start
            if cut == previous_cut:
                break
        ready, self.pending = text[:cut], text[cut:]
        for secret in self.secrets:
            ready = ready.replace(secret, "[redacted]")
        return ready.encode("utf-8")


def _bash_progress_line(
    stdout_chunks: _BashOutput,
    stderr_chunks: _BashOutput,
    context: ToolContext | None = None,
) -> str:
    """One short status line for a running background command.

    Reports the LAST non-empty line the command printed, which for the work
    that gets backgrounded (builds, training loops, terraform, pollers) is the
    step it is currently on. Bounded hard: this is written on every poll tick
    and read by a renderer, so an unbounded line from a command printing a
    megabyte without newlines must not become a per-frame cost.
    """
    tail = b"".join(_tail_chunks(stdout_chunks, 1024) + _tail_chunks(stderr_chunks, 1024)).decode(
        "utf-8", errors="replace"
    )
    for line in reversed(tail.splitlines()):
        if line.strip():
            return _redact_tool_text(line.strip()[:200], context)
    return "running"


def _redact_tool_text(text: str, context: ToolContext | None) -> str:
    """Strip stored session-credential values out of tool output.

    The LOOP's ``redact_tool_result`` hook is the model-visible choke point;
    this stays for the UIs that read live output BEFORE the result exists
    (bash stream updates, background-job peek, the abort receipt). A command
    that ``echo``s ``$GITHUB_TOKEN`` would otherwise paint the secret while
    the command is still running.
    """
    store = context.variables if context is not None else None
    redact = getattr(store, "redact", None)
    if callable(redact):
        redacted = redact(text)
        return redacted if isinstance(redacted, str) else text
    return text


def _bash_output_summary(stdout: str, stderr: str) -> str:
    """The shared 'stdout/stderr' body used by updates and the final result."""
    parts = [
        f"--- stdout ---\n{stdout}" if stdout else "--- stdout ---\n(empty)",
        f"--- stderr ---\n{stderr}" if stderr else "--- stderr ---\n(empty)",
    ]
    return "\n".join(parts)


def _stream_budgets(stdout: str, stderr: str, budget: int, failed: bool) -> tuple[int, int]:
    """Split ``budget`` chars between stdout and stderr, stderr first.

    Truncation's real risk is not lost bytes, it is a model that can no longer
    see why something failed and starts guessing or re-running — which costs
    more than the truncation saved. So the diagnostic stream gets first claim:
    stderr takes whatever it needs up to a share of the budget, and stdout
    takes the rest. The share rises to 3/4 when the command exited non-zero,
    because on a failing run stderr IS the answer, and stays at 1/2 otherwise
    so a chatty-but-harmless stderr (progress bars, deprecation warnings)
    cannot crowd out the stdout the model actually asked for.

    Both sides keep at least one char so neither stream can vanish entirely
    without a marker explaining that it was there.
    """
    stderr_share = (budget * 3) // 4 if failed else budget // 2
    stderr_budget = min(len(stderr), stderr_share)
    stdout_budget = budget - stderr_budget
    # A huge stderr on a failing command can leave stdout nothing; a huge
    # stdout with tiny stderr leaves stderr nothing. Neither is acceptable as
    # a zero, because a zero budget renders as an empty stream rather than a
    # truncated one and the model reads it as "there was no output".
    return max(stdout_budget, 1), max(stderr_budget, 1)


#: Bytes of accumulated output the bash live-update snapshot is built from,
#: counted from the END. The card's live view ingests only the last
#: :data:`~local_operator.tui.widgets.tool_card.LIVE_INGEST_CHARS` (64 KB) of
#: the payload anyway, so a producer snapshot beyond one tail is work whose
#: result is dropped unrendered. Before this bound the emit was O(total
#: output) per 500 ms tick — join, decode and redact of EVERYTHING the command
#: has ever printed, twice a second, for a command that has printed megabytes —
#: which is the tool-call freeze the operator reported. Kept one power of two
#: above the ingest bound so the banner framing and the redaction of a secret
#: that straddles the cut cannot shrink the visible tail below what the card
#: expects. The final result formats the bounded retained head/tail once,
#: off-loop; output beyond the existing spill cap is explicitly marked partial.
_EMIT_SNAPSHOT_BYTES = 128 * 1024


def _tail_chunks(chunks: list[bytes] | _BashOutput, budget: int) -> list[bytes]:
    """The trailing ``budget`` bytes of ``chunks`` without a full join.

    Walking from the end keeps the work proportional to the BUDGET rather
    than to the accumulated output: a command that has printed 40 MB costs
    the same slice as one that has printed 40 KB. Chunks are immutable
    records of what the pipe reader delivered, so slicing the straddling
    one copies at most ``budget`` bytes once — still O(budget).
    """
    if isinstance(chunks, _BashOutput):
        # Do not materialize the retained 2 MiB head for a 128 KiB live tail.
        # Only a stream shorter than the requested tail needs head bytes.
        head_needed = max(budget - chunks.tail_bytes, 0)
        head = [bytes(chunks.head[-head_needed:])] if head_needed else []
        chunks = [*head, *chunks.tail]
    taken: list[bytes] = []
    remaining = budget
    for chunk in reversed(chunks):
        if remaining <= 0:
            break
        taken.append(chunk[-remaining:] if len(chunk) > remaining else chunk)
        remaining -= len(taken[-1])
    taken.reverse()
    return taken


@_guard("bash")
async def execute_bash(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Run a shell command non-interactively and capture its output.

    Output is read incrementally by per-stream reader tasks (kept referenced,
    so they are never orphaned): accumulated output streams to ``on_update``
    roughly every 500 ms while the command runs, and partial output survives
    both abort and timeout.
    """
    try:
        params = BashParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "bash", exc)
    if not params.command.strip():
        return _error(tool_call_id, "bash", "command must be a non-empty string")
    # Approval for write/exec tiers is the LOOP's gate (it fires after
    # tool_execution_start so the UI shows the pending call). A second gate
    # here made the user answer twice per action, with the tier name rendered
    # as the tool. Read-tier outside-workspace escalations still use
    # _check_approval in execute_read/execute_grep.

    # Pre-aborted signal: never spawn a child there is no intention to run.
    if signal is not None and signal.aborted:
        return _error(
            tool_call_id,
            "bash",
            f"aborted ({signal.reason or 'aborted'}): {params.command}",
        )

    env = os.environ.copy()
    env.update(NON_INTERACTIVE_ENV)
    # Session credentials ride the child environment so the agent can USE a
    # secret it can never READ. Injected here rather than advertised as a
    # bash ``env`` argument: a model-authored env map would have to carry
    # the value (or a placeholder we do not have), which is the leak this
    # store exists to prevent.
    store = context.variables if context is not None else None
    credential_env = getattr(store, "credential_env", None)
    extra = credential_env() if callable(credential_env) else None
    if isinstance(extra, dict):
        extra = {str(name): str(value) for name, value in extra.items()}
        env.update(extra)

    process = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        params.command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=_safe_cwd(context),
        env=env,
        start_new_session=True,
    )

    # Record this group in the owner's process-group ledger so a HARD death of
    # THIS lop process (SIGKILL from cmux, OOM, crash) — the one stop path with
    # no in-process code to run _kill() — can still be reaped at the next
    # startup. start_new_session already detached the group from the terminal,
    # so a hard-dead owner would otherwise leak it to init forever. Best-effort
    # and owner-liveness-only by construction: the group is reaped iff THIS
    # process is dead, never by runtime/CPU/idle, so a legit long-runner (a 10h
    # trainer) is safe while its session lives. See tools/group_reaper.py.
    # suppress(Exception): os.getpgid is POSIX-only (absent on Windows), and
    # registration must never be the reason a command fails to run. The pgid is
    # held so the clean-death paths below can unregister the exact line.
    spawned_pgid: int | None = None
    with contextlib.suppress(Exception):
        spawned_pgid = os.getpgid(process.pid)
        group_reaper.register_group(spawned_pgid, params.command)

    def _unregister_group() -> None:
        # Drop this group's ledger line once it is confirmed dead, so a clean
        # run leaves nothing for the startup sweep to consider and a long host
        # session's ledger cannot grow without bound. Best-effort; a miss is
        # harmless (the sweep finds the leader gone and drops the line anyway).
        if spawned_pgid is None:
            return
        with contextlib.suppress(Exception):
            group_reaper.unregister_group(spawned_pgid)

    stdout_chunks = _BashOutput()
    stderr_chunks = _BashOutput()
    # Set once the command is owned by a background job, so the pipe readers
    # know where to mirror output for `jobs(op="peek")`. Held in a mutable cell
    # rather than captured by value because the readers start BEFORE the job id
    # exists on the steering-detach path: the command is already running when
    # the interrupt arrives, and re-creating the readers at that point would
    # race the drain and lose whatever is in flight.
    live_job: dict[str, Any] = {"id": None, "jobs": None}

    def _mirror(chunk: bytes) -> None:
        """Publish a freshly-read chunk to the job's peekable tail."""
        job_id = live_job["id"]
        manager = live_job["jobs"]
        if job_id is None or manager is None:
            return
        appender = getattr(manager, "append_output", None)
        if appender is None:
            # Third-party embedders may supply a manager predating live output;
            # peek degrades to "no output recorded" rather than breaking the run.
            return
        appender(
            job_id,
            _redact_tool_text(chunk.decode("utf-8", errors="replace"), context),
        )

    async def _pump(stream: asyncio.StreamReader | None, sink: _BashOutput) -> None:
        # Both pipes were requested at spawn, so neither is ever None here;
        # the guard keeps the reader honest instead of asserting.
        if stream is None:
            return
        redactor = _PipeRedactor(extra if isinstance(extra, dict) else {})
        try:
            while True:
                chunk = await stream.read(65536)
                if not chunk:
                    break
                safe = redactor.feed(chunk)
                sink.append(safe)
                _mirror(safe)
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            safe = redactor.feed(b"", final=True)
            sink.append(safe)
            _mirror(safe)

    # Hold the tasks ourselves so the readers are never abandoned mid-run.
    stdout_task = asyncio.create_task(_pump(process.stdout, stdout_chunks))
    stderr_task = asyncio.create_task(_pump(process.stderr, stderr_chunks))
    readers = (stdout_task, stderr_task)

    def _kill() -> None:
        # Kill the whole session group so children (sh -c spawns) die too.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(process.pid), signal_module.SIGKILL)

    def _emit_update() -> None:
        if on_update is None:
            return
        # Bounded to the live-view tail (see _EMIT_SNAPSHOT_BYTES): the card
        # renders at most 64 KB of this snapshot, so building more is pure
        # per-tick cost that scales with the command's TOTAL output — the
        # reported freeze. The tail is taken from the raw chunk list so the
        # join itself is bounded too; a chunk straddling the cut is sliced,
        # which can split a multi-byte character, and the errors="replace"
        # decode below is exactly the repair that path already relies on for
        # arbitrary command bytes.
        stdout = _redact_tool_text(
            b"".join(_tail_chunks(stdout_chunks, _EMIT_SNAPSHOT_BYTES)).decode(
                "utf-8", errors="replace"
            ),
            context,
        )
        stderr = _redact_tool_text(
            b"".join(_tail_chunks(stderr_chunks, _EMIT_SNAPSHOT_BYTES)).decode(
                "utf-8", errors="replace"
            ),
            context,
        )
        on_update(
            AgentToolUpdate(
                content=[TextContent(text=_bash_output_summary(stdout, stderr))],
                details={"tool_name": "bash", "running": True},
            )
        )

    loop = asyncio.get_running_loop()
    deadline = loop.time() + params.timeout
    wait_task = asyncio.create_task(process.wait())
    abort_waiter = asyncio.create_task(signal.wait()) if signal is not None else None

    timed_out = False
    aborted = False
    next_update = loop.time() + 0.5

    def _detach_to_job(jobs: Any, headline: str) -> ToolResult:
        """Hand the running process to a background job and return its id.

        Shared by the two ways a command stops being awaited: the caller asked
        for ``background=True`` up front, and steering interrupted a foreground
        call. Both want identical ownership semantics — the process keeps
        running in its own session group, the pipe readers stay alive so a full
        pipe cannot block it, output keeps flowing to the peek buffer, and the
        original timeout budget still applies — so they share one
        implementation rather than two that drift.
        """
        partial = _bash_output_summary(
            _redact_tool_text(
                b"".join(_tail_chunks(stdout_chunks, TOOL_OUTPUT_LIMIT_CHARS)).decode(
                    "utf-8", errors="replace"
                ),
                context,
            ),
            _redact_tool_text(
                b"".join(_tail_chunks(stderr_chunks, TOOL_OUTPUT_LIMIT_CHARS)).decode(
                    "utf-8", errors="replace"
                ),
                context,
            ),
        )
        wait_task.cancel()
        if abort_waiter is not None and not abort_waiter.done():
            abort_waiter.cancel()
        command = params.command
        remaining_timeout = max(deadline - loop.time(), 0.0)

        async def _detached(job_id: str, job_signal: Any, report_progress: Any) -> str:
            # Owns the process from here: waits with the ORIGINAL timeout
            # budget, keeps the readers alive to drain the pipes, and reports
            # the exit status + bounded output as the job result.
            del job_id
            timed_out_bg = False
            cancelled_bg = False
            bg_deadline = asyncio.get_running_loop().time() + remaining_timeout
            bg_wait = asyncio.create_task(process.wait())

            async def cleanup(*, kill: bool) -> None:
                """Kill/reap the whole process group and close EVERY owner.

                AsyncJobManager.cancel both aborts the job signal and cancels
                this runner immediately. Without a cancellation handler, that
                cancellation can land inside the wait below and settle the job
                while the start-new-session child + pipe readers keep running
                untracked. Cleanup is bounded but exhaustive: process group,
                waiter, both readers, transport.
                """
                if kill and process.returncode is None:
                    _kill()
                if not bg_wait.done():
                    with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                        await asyncio.wait_for(asyncio.shield(bg_wait), timeout=1.0)
                if process.returncode is None:
                    _kill()
                    with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                        await asyncio.wait_for(process.wait(), timeout=1.0)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        asyncio.gather(stdout_task, stderr_task, return_exceptions=True),
                        timeout=0.5,
                    )
                for reader in readers:
                    if not reader.done():
                        reader.cancel()
                if readers:
                    await asyncio.gather(*readers, return_exceptions=True)
                if not bg_wait.done():
                    bg_wait.cancel()
                    await asyncio.gather(bg_wait, return_exceptions=True)
                transport = getattr(process, "_transport", None)
                if transport is not None:
                    transport.close()
                # Clean group death: drop its ledger line (see _unregister_group).
                _unregister_group()

            try:
                while not bg_wait.done():
                    if job_signal is not None and job_signal.aborted:
                        cancelled_bg = True
                        break
                    if asyncio.get_running_loop().time() > bg_deadline:
                        timed_out_bg = True
                        break
                    await asyncio.wait({bg_wait}, timeout=0.25)
                    # The status line a human reads in the TUI while the job
                    # runs. Deliberately a heartbeat and not the output itself:
                    # the OUTPUT has a dedicated bounded channel (the peek
                    # tail), and mirroring it into a field every renderer
                    # repaints per frame would pay for it many times over.
                    report_progress(_bash_progress_line(stdout_chunks, stderr_chunks, context))
                await cleanup(kill=cancelled_bg or timed_out_bg)
            except asyncio.CancelledError:
                # Manager cancellation is deliberately immediate. Convert it
                # into process cleanup first, then preserve cancellation so the
                # job row settles as cancelled rather than completed/failed.
                await cleanup(kill=True)
                raise

            out, err = await asyncio.to_thread(_decode_chunks, stdout_chunks, stderr_chunks)
            out = _redact_tool_text(out, context)
            err = _redact_tool_text(err, context)
            code = process.returncode if process.returncode is not None else -1
            head = f"TIMEOUT after {params.timeout}s (process killed)" if timed_out_bg else ""
            if cancelled_bg:
                head = "CANCELLED (process killed)"
            out, err, footer, _spill_details = await asyncio.to_thread(
                _bash_oversized_streams,
                out,
                err,
                TOOL_OUTPUT_LIMIT_CHARS - 2 * len(BASH_TRUNCATION_MARKER),
                code != 0 or timed_out_bg,
                context,
                not (stdout_chunks.omitted_bytes or stderr_chunks.omitted_bytes),
            )
            summary = _bash_output_summary(out, err) + footer
            return "\n".join(part for part in (head, f"exit code: {code}", summary) if part)

        def _kill_unstarted() -> None:
            """Teardown for a cancel that lands before the runner is entered.

            The process is spawned before ``register``, and ``register`` only
            SCHEDULES the runner — so a cancel (or a session ``dispose``) in
            the same event-loop turn settles the row without ``_detached``
            ever running, leaving the process group alive and reparented to
            init with nothing tracking it. The manager drops this hook the
            instant the runner starts, so the group is never killed twice.
            """
            _kill()
            for reader in readers:
                if not reader.done():
                    reader.cancel()

        try:
            bg_job_id = cast(Any, jobs).register(
                "bash",
                f"bash: {command[:60]}",
                _detached,
                # Unowned ON PURPOSE, matching how ``run_subagent`` registers
                # its ``task`` jobs. An owner is only useful with a registered
                # delivery sink, and nothing in this codebase calls
                # ``register_delivery_sink`` — so setting one guarantees the
                # opposite of what it looks like: the completion is
                # DEAD-LETTERED ("no live sink for owner ...") and the caller
                # is never told its background job finished, which is the whole
                # point of running it detached. Revisit together with a sink
                # implementation, not before.
                owner_id=None,
                on_cancel=_kill_unstarted,
            )
        except Exception:  # noqa: BLE001 — no manager slot: kill, don't leak
            _kill()
            raise
        # Point the pipe readers at the job BEFORE reporting it, and seed the
        # tail with what the foreground phase already collected, so a peek
        # covers the command's whole life rather than starting from whenever it
        # happened to be backgrounded.
        live_job["jobs"] = jobs
        live_job["id"] = bg_job_id
        appender = getattr(jobs, "append_output", None)
        if appender is not None:
            already = b"".join(
                _tail_chunks(stdout_chunks, _EMIT_SNAPSHOT_BYTES)
                + _tail_chunks(stderr_chunks, _EMIT_SNAPSHOT_BYTES)
            ).decode("utf-8", errors="replace")
            if already:
                appender(bg_job_id, _redact_tool_text(already, context))
        return _text(
            tool_call_id,
            "bash",
            f"job {bg_job_id}: {headline}\ncommand: {command}\n{partial}",
            details={"job_id": bg_job_id, "backgrounded": True},
        )

    if params.background:
        # Deliberate backgrounding: hand the command straight to a job and give
        # the model its id, so the very next call can peek at or cancel it.
        # Without this the ONLY route to a background job was a steering
        # interrupt — an accident the model cannot ask for — which left long
        # work (training, terraform, pipeline polls) with no option but to
        # block a turn for its whole duration.
        background_jobs = context.jobs if context is not None else None
        if background_jobs is None:
            # Tear down everything this call created before refusing. The
            # readers are not the only owners: ``wait_task`` and the abort
            # waiter were created above and would outlive the refusal as
            # pending tasks holding the process handle.
            _kill()
            wait_task.cancel()
            if abort_waiter is not None and not abort_waiter.done():
                abort_waiter.cancel()
            for reader in readers:
                reader.cancel()
            return _error(
                tool_call_id,
                "bash",
                "background=true needs a job manager, which this session has "
                "not attached; re-run without background.",
            )
        return _detach_to_job(
            background_jobs,
            "started in the background. Use jobs(op='peek', job_id=..., "
            "since=<seq>) for new output and jobs(op='cancel') to stop it; "
            "its result auto-delivers when it finishes.",
        )

    try:
        while True:
            # Pipe EOF can precede exit (a child may close both descriptors and
            # continue computing). Race only unfinished tasks or FIRST_COMPLETED
            # would repeatedly wake on the same EOF and spin. ALL_COMPLETED, in
            # contrast, waits the full poll interval for a never-fired abort.
            waiters: list[asyncio.Task[object]] = [
                task for task in (wait_task, stdout_task, stderr_task) if not task.done()
            ]
            if abort_waiter is not None:
                waiters.append(abort_waiter)
            if wait_task.done():
                break  # finished already — never misreport as timeout
            remaining = deadline - loop.time()
            if remaining <= 0:
                timed_out = True
                _kill()
                break
            done, _pending = await asyncio.wait(
                waiters, timeout=min(0.25, remaining), return_when=asyncio.FIRST_COMPLETED
            )
            if wait_task in done:
                break
            if abort_waiter is not None and abort_waiter in done:
                aborted = True
                _kill()
                break
            if loop.time() >= next_update:
                _emit_update()
                next_update = loop.time() + 0.5
    except asyncio.CancelledError:
        # Steering interrupted the tool task (the loop cancels interruptible
        # tools at its 0.25s poll). Killing the process here used to (a)
        # destroy minutes of a long build on a one-line user aside and (b)
        # leak the child anyway when cancellation raced the kill. Instead the
        # command DETACHES: it keeps running (it was spawned start_new_session
        # so it survives its own process group), its readers keep draining so
        # a full pipe cannot block it, and it is tracked as a background job
        # whose completion auto-delivers when the session is idle. A REAL
        # abort (Ctrl+C, jobs cancel) still kills: that is a stop, not a
        # redirect.
        if signal is not None and signal.aborted:
            _kill()
            raise
        jobs = context.jobs if context is not None else None
        if jobs is None:
            # No job manager to own a detached child: kill rather than leak.
            _kill()
            raise
        return _detach_to_job(
            jobs,
            "steering interrupted; the command continues in the background. "
            "Use jobs(op='peek', job_id=..., since=<seq>) for new output and "
            "jobs(op='cancel') to stop it; its result auto-delivers when the "
            "session is idle.",
        )

    # Bounded drain: the kill above EOFs both pipes; give the readers 250 ms
    # to consume what is already buffered so partial output survives.
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(asyncio.gather(*readers, return_exceptions=True), timeout=0.25)
    for task in readers:
        if not task.done():
            task.cancel()
            with contextlib.suppress(BaseException):
                await task

    # Reap the process and release the transport so no ResourceWarning fires.
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(process.wait(), timeout=1.0)
    if process.returncode is None:
        _kill()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=1.0)
    transport = getattr(process, "_transport", None)
    if transport is not None:
        transport.close()
    # Normal-exit reap complete: the group is dead, so drop its ledger line.
    _unregister_group()

    if abort_waiter is not None and not abort_waiter.done():
        abort_waiter.cancel()
        with contextlib.suppress(BaseException):
            await abort_waiter

    if aborted:
        partial = await asyncio.to_thread(_bash_partial_summary, stdout_chunks, stderr_chunks)
        return _error(
            tool_call_id,
            "bash",
            f"aborted ({(signal.reason or 'aborted') if signal else 'aborted'}): "
            f"{params.command}\n{_redact_tool_text(partial, context)}",
        )

    # Decoding and, for oversized output, spilling/eliding run in a thread:
    # a command that printed megabytes turns this tail into a multi-MB
    # decode, a multi-MB join, a disk write of the spill and string slicing
    # to elide it — all synchronous, all on the loop that renders the TUI,
    # and the reason a batch of concurrent bash calls used to freeze the
    # frame at the moment they finished together.
    stdout_raw, stderr_raw = await asyncio.to_thread(_decode_chunks, stdout_chunks, stderr_chunks)
    stdout_raw = _redact_tool_text(stdout_raw, context)
    stderr_raw = _redact_tool_text(stderr_raw, context)
    return_code = process.returncode if process.returncode is not None else -1

    # Both streams may end up carrying a marker, so reserve room for two.
    budget = TOOL_OUTPUT_LIMIT_CHARS - 2 * len(BASH_TRUNCATION_MARKER)
    spill_details: dict[str, Any] | None = None
    if len(stdout_raw) + len(stderr_raw) > budget:
        stdout, stderr, footer, spill_details = await asyncio.to_thread(
            _bash_oversized_streams,
            stdout_raw,
            stderr_raw,
            budget,
            return_code != 0 or timed_out,
            context,
            not (stdout_chunks.omitted_bytes or stderr_chunks.omitted_bytes),
        )
    else:
        stdout, stderr = stdout_raw, stderr_raw
        footer = ""

    parts = [f"exit code: {return_code}", _bash_output_summary(stdout, stderr)]
    if timed_out:
        parts.insert(0, f"TIMEOUT after {params.timeout}s (process killed)")
    return _text(tool_call_id, "bash", "\n".join(parts) + footer, details=spill_details)


def _bash_partial_summary(stdout_chunks: _BashOutput, stderr_chunks: _BashOutput) -> str:
    """Both streams' partial text under the abort receipt's framing."""
    return _bash_output_summary(
        truncate_output(stdout_chunks.decode()),
        truncate_output(stderr_chunks.decode()),
    )


def _decode_chunks(stdout_chunks: _BashOutput, stderr_chunks: _BashOutput) -> tuple[str, str]:
    """Join and decode both captured streams off the event loop."""
    return (
        stdout_chunks.decode(),
        stderr_chunks.decode(),
    )


def _bash_oversized_streams(
    stdout_raw: str,
    stderr_raw: str,
    budget: int,
    failed: bool,
    context: ToolContext | None,
    source_complete: bool = True,
) -> tuple[str, str, str, dict[str, Any] | None]:
    """The oversized-output tail of ``bash``: spill once, elide both streams.

    Synchronous by design — ``asyncio.to_thread`` is the only caller. The
    framing decisions moved here verbatim from the loop-bound block so the
    bytes a model reads do not change with the thread they are built on.

    ONE spill for the WHOLE transcript, in exactly the framing the model
    already sees. Spilling the two streams separately would hand out two
    handles for one command and make "line 900" ambiguous; spilling a
    differently-framed copy would make the footer's line numbers point
    somewhere other than where they resolve.
    """
    combined = _bash_output_summary(stdout_raw, stderr_raw)
    if len(stdout_raw) + len(stderr_raw) <= budget and source_complete:
        return stdout_raw, stderr_raw, "", None
    meta = get_store().write(
        combined,
        tool_name="bash",
        session_id=(context.session_id if context else "") or "",
        source_complete=source_complete,
    )
    stdout_budget, stderr_budget = _stream_budgets(stdout_raw, stderr_raw, budget, failed=failed)
    spill_details: dict[str, Any] | None = None
    footer = ""
    if meta is None:
        stdout = truncate_output(stdout_raw, stdout_budget)
        stderr = truncate_output(stderr_raw, stderr_budget)
    elif not meta.complete:
        # The stored copy has its own head/tail cap. Original-stream line
        # offsets no longer address that copy; offer its real first page and
        # search, never a plausible-looking out-of-range expansion command.
        spill_details = {"spill": _spill_detail(meta)}
        stdout = truncate_output(stdout_raw, stdout_budget)
        stderr = truncate_output(stderr_raw, stderr_budget)
        footer = _spill_footer(meta)
    else:
        spill_details = {"spill": _spill_detail(meta)}
        # Offsets map each stream's local line numbers onto the combined
        # transcript the handle serves: line 1 is the '--- stdout ---'
        # banner, and the stderr banner sits after the whole stdout block.
        stdout_lines = len(stdout_raw.splitlines()) if stdout_raw else 1
        stdout, stdout_span = _elide_inline(stdout_raw, stdout_budget, offset=1)
        stderr, stderr_span = _elide_inline(stderr_raw, stderr_budget, offset=2 + stdout_lines)
        # Suggest the STDERR gap when the command failed and stderr is the
        # stream that lost content: on a failing run that is where the
        # model needs to look, and a footer that points at the stdout gap
        # instead sends it to the least useful region of the output.
        suggested = (stderr_span if failed and stderr_span else None) or stdout_span
        footer = _spill_footer(meta, suggested)
    return stdout, stderr, footer, spill_details


def build_bash_tool() -> AgentTool:
    return AgentTool(
        name="bash",
        label="Shell",
        describe_approval=_describe_shell_approval,
        description=("Run a shell command and return its exit code, stdout and stderr."),
        parameters=BashParams.model_json_schema(),
        approval_tier="exec",
        # bash runs shared when non-pty; models batch independent
        # commands, and exclusive would serialize the common case.
        concurrency="shared",
        interruptible=True,
        execute=execute_bash,
    )


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------


class ReadParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        description=(
            "File path (absolute or relative to the working directory), or an "
            "internal URL: skill://<name> (its reference files via "
            "skill://<name>/<relpath>, listed at the end of the skill body — "
            "never via a raw filesystem path), or spill://<id> to expand an "
            "output that was truncated (append '?q=<regex>' to search inside "
            "it instead of paging through it)."
        )
    )
    range: str | None = Field(
        default=None,
        description=(
            "Optional 1-based inclusive line range 'start-end' (e.g. '10-40') "
            "or 'start-' to read to the end. Applies to files and to "
            "spill:// handles; on a spill search ('?q=') it selects which "
            "MATCHES to return, not which lines. Ignored for other internal URLs."
        ),
    )
    raw: bool = Field(
        default=False,
        description=(
            "Return the verbatim text even where a structural summary is the "
            "default (Python files read without a range)."
        ),
    )


_LINE_RANGE_RE = re.compile(r"^(\d+)\s*-\s*(\d+)?$")


def _parse_line_range(spec: str) -> tuple[int, int | None]:
    match = _LINE_RANGE_RE.match(spec.strip())
    if not match:
        raise ValueError(f"invalid line range '{spec}' (expected 'start-end' or 'start-')")
    start = int(match.group(1))
    if start < 1:
        raise ValueError(f"invalid line range '{spec}': start must be >= 1")
    end = int(match.group(2)) if match.group(2) else None
    if end is not None and end < start:
        raise ValueError(f"invalid line range '{spec}': end must be >= start")
    return start, end


def _number_lines(lines: list[str], start: int) -> str:
    width = len(str(start + len(lines) - 1))
    return "\n".join(f"{start + i:>{width}}| {line}" for i, line in enumerate(lines))


def _clamp_file_body(body: str, path: Path, start: int, total: int) -> str:
    """Hold one ``read`` result inside the char budget.

    A file needs no spill entry: the file IS the store, it is already on disk,
    and the footer names a range on the same path. Copying it into the spill
    directory would double the bytes to recover something ``read`` can already
    address — the exact unbounded-retention mistake this work exists to avoid.

    Only the HEAD is kept here, unlike command output. A file is random-access
    by line and the model chose the offset, so the useful continuation is
    "carry on from where this stopped"; splicing in a tail would break the
    contiguity that makes a numbered listing readable.
    """
    if len(body) <= READ_OUTPUT_LIMIT_CHARS:
        return body
    clipped = body[:READ_OUTPUT_LIMIT_CHARS]
    cut = clipped.rfind("\n")
    if cut > 0:
        clipped = clipped[:cut]
    next_line = start + len(clipped.splitlines())
    return (
        f"{clipped}\n\n[truncated at {READ_OUTPUT_LIMIT_CHARS} chars; "
        f"{total - next_line + 1} of {total} lines not shown. Continue with "
        f'read(path="{path}", range="{next_line}-{next_line + 200}") '
        f"or narrow with grep]"
    )


def _joined_capped_list_body(
    full_items: list[str],
    shown_items: list[str],
    tool_name: str,
    context: ToolContext | None,
) -> tuple[str, dict[str, Any] | None]:
    """``_capped_list_body`` over two item lists, joined off the loop.

    The joins live in here — and not in the caller's ``to_thread`` argument
    list, where Python would evaluate them on the event loop before the call
    — because joining thousands of match lines is exactly the multi-MB
    string work this exists to move.
    """
    return _capped_list_body("\n".join(full_items), "\n".join(shown_items), tool_name, context)


def _spilled_list_body(
    full_lines: list[str],
    shown_text: str,
    tool_name: str,
    context: ToolContext | None,
) -> tuple[str, dict[str, Any] | None]:
    """Join a spill's full line set while preserving an authored shown body.

    Grep's current renderer adds context lines and paging instructions to
    ``shown_text``; only the recoverable match lines belong in the spill.
    Keeping the join here means the caller's ``to_thread`` truly moves the
    multi-megabyte work rather than eagerly evaluating it on the event loop.
    """
    return _capped_list_body("\n".join(full_lines), shown_text, tool_name, context)


def _capped_list_body(
    full: str, shown: str, tool_name: str, context: ToolContext | None
) -> tuple[str, dict[str, Any] | None]:
    """Body + spill details for a list result that has TWO caps on it.

    ``glob`` and ``grep`` cap by item count first (500 paths, 200 matches) and
    then still have to fit the char budget — 200 grep hits on long lines is
    comfortably over it. Both caps discard content, so the handle is written
    from ``full`` (everything found) while only ``shown`` (the count-capped
    prefix) is measured against the budget. Writing the spill from ``shown``
    would make the handle a copy of what the model can already see, which is
    the one thing an expansion path must never be.
    """
    if full == shown and len(shown) <= TOOL_OUTPUT_LIMIT_CHARS:
        return shown, None
    meta = _spill(full, tool_name, context)
    if meta is None:
        return truncate_output(shown, TOOL_OUTPUT_LIMIT_CHARS), None
    if len(shown) <= TOOL_OUTPUT_LIMIT_CHARS:
        # Fits the prompt, but the count cap still hid entries. Point at the
        # rest explicitly rather than leaving "(capped at N)" as a dead end.
        hidden_from = len(shown.splitlines()) + 1
        return shown + _spill_footer(meta, (hidden_from, meta.lines)), {
            "spill": _spill_detail(meta)
        }
    body, span = _elide_inline(shown, TOOL_OUTPUT_LIMIT_CHARS)
    return body + _spill_footer(meta, span), {"spill": _spill_detail(meta)}


def _read_spill(tool_call_id: str, target: str, range_spec: str | None) -> ToolResult:
    """Serve a ``spill://`` handle: a line range, or a regex search inside it.

    Routed here BEFORE the host's internal-URL resolver rather than registered
    as another resolver, because a resolver returns one opaque string and this
    path is the whole point of the store: it must apply a RANGE. Reusing
    ``read``'s existing range parsing and line numbering is also what keeps
    this from becoming a second convention — an agent addresses a spilled
    output exactly the way it addresses a file.
    """
    ref = parse_handle(target)
    if ref is None:
        return _error(
            tool_call_id,
            "read",
            f"Malformed spill handle '{target}'. Expected "
            f"'{SPILL_SCHEME}<32 hex chars>' optionally followed by '?q=<regex>'.",
        )
    store = get_store()
    meta = store.stat(ref.handle)
    if meta is None:
        # A bounded store evicts, so this is an ordinary outcome and the
        # message says what to do about it rather than reading as a fault.
        return _error(
            tool_call_id,
            "read",
            f"Spilled output {ref.handle} is no longer available (the store is "
            "size-bounded and evicts least-recently-used entries). Re-run the "
            "command that produced it if you still need the full output.",
        )

    if ref.query:
        return _search_spill(tool_call_id, store, ref, meta, range_spec)

    start, end = 1, None
    if range_spec:
        try:
            start, end = _parse_line_range(range_spec)
        except ValueError as exc:
            return _error(tool_call_id, "read", str(exc))
    result = store.read_lines(ref.handle, start, end)
    if result is None:
        return _error(tool_call_id, "read", f"Spilled output {ref.handle} could not be read.")
    selected, total = result
    details: dict[str, Any] = {"url": ref.handle, "range": range_spec or "full"}
    if not selected:
        return _text(
            tool_call_id,
            "read",
            f"(range {range_spec} is beyond the end of {ref.handle}; it has {total} lines)",
            useless=True,
            details={**details, "useless": True},
        )
    body = _number_lines(selected, start)
    # An unranged read of a spill is itself capped: expanding "the whole
    # thing" must not undo the truncation that created the handle. The footer
    # points back at the same handle with a concrete next range, so a model
    # that really does want to walk the output can, one bounded page at a time.
    if len(body) > READ_OUTPUT_LIMIT_CHARS:
        head, tail = _clip_head_tail(body, READ_OUTPUT_LIMIT_CHARS - len(BASH_TRUNCATION_MARKER))
        shown = len(head.splitlines())
        body = (
            head
            + BASH_TRUNCATION_MARKER
            + tail
            + f"\n[this page was itself truncated. Continue with "
            f'read(path="{ref.handle}", range="{start + shown}-{start + shown + 200}") '
            f'or narrow first with read(path="{ref.handle}?q=<regex>")]'
        )
    header = f"{ref.handle} — lines {start}-{start + len(selected) - 1} of {total}"
    if not meta.complete:
        header += " (stored copy is head+tail of an over-cap output)"
    return _text(tool_call_id, "read", f"{header}\n{body}", details=details)


def _search_spill(
    tool_call_id: str,
    store: SpillStore,
    ref: SpillRef,
    meta: SpillMeta,
    range_spec: str | None,
) -> ToolResult:
    """``?q=<regex>`` over one spilled output: line numbers, then a range.

    This is the difference between expansion being usable and being a slower
    way to re-run the command. Paging a 4,000-line pytest log at ~2k tokens a
    page to find one traceback costs more than the truncation ever saved;
    finding the line number for ~200 tokens and reading 40 lines around it
    costs almost nothing.
    """
    try:
        found = store.search(ref.handle, ref.query, SPILL_SEARCH_MATCH_LIMIT)
    except re.error as exc:
        return _error(tool_call_id, "read", f"invalid regex '{ref.query}': {exc}")
    if found is None:
        return _error(tool_call_id, "read", f"Spilled output {ref.handle} could not be read.")
    matches, total_matches, total_lines = found
    if range_spec:
        # Documented in the schema: on a search the range pages through
        # MATCHES. Slicing lines here instead would silently return nothing
        # whenever the matches fell outside the requested line window.
        try:
            start, end = _parse_line_range(range_spec)
        except ValueError as exc:
            return _error(tool_call_id, "read", str(exc))
        matches = matches[start - 1 : end]
    details = {"url": f"{ref.handle}?q={ref.query}"}
    if not matches:
        return _text(
            tool_call_id,
            "read",
            f"No lines match '{ref.query}' in {ref.handle} ({total_lines} lines searched).",
            useless=True,
            details={**details, "useless": True},
        )
    width = len(str(matches[-1][0]))
    body = "\n".join(f"{number:>{width}}| {line}" for number, line in matches)
    header = (
        f"{len(matches)} of {total_matches} match(es) for '{ref.query}' in "
        f"{ref.handle} ({total_lines} lines)"
    )
    if total_matches > len(matches):
        header += "; 'range' pages through matches"
    footer = (
        f'\n[read around a hit with read(path="{ref.handle}", '
        f'range="{max(matches[0][0] - 10, 1)}-{matches[0][0] + 30}")]'
    )
    return _text(tool_call_id, "read", f"{header}:\n{body}{footer}", details=details)


# ---------------------------------------------------------------------------
# Python structural summaries
# ---------------------------------------------------------------------------

#: Below this many lines a raw read is already cheap and the summary's
# overhead (footer, symbol grammar) buys nothing — the default stays raw.
PYTHON_SUMMARY_MIN_LINES = 80
#: Hard cap on symbol lines in one summary. A generated module with
# thousands of definitions would otherwise turn the summary into the very
# blob it exists to avoid; past the cap the tail is elided with a count.
PYTHON_SUMMARY_MAX_SYMBOLS = 500


def _format_args(args: ast.arguments) -> str:
    """``name: Ann`` parts in signature order, stdlib-only. Defaults are
    interleaved onto their parameters (pydantic stores them positionally
    against the tail of the arg list, so a naive arg walk silently drops
    every ``= default`` from the summary)."""
    plain = list(getattr(args, "posonlyargs", [])) + list(args.args)
    defaults: list[ast.expr] = list(args.defaults)
    offset = len(plain) - len(defaults)
    parts: list[str] = []
    posonly = len(getattr(args, "posonlyargs", []))
    for i, arg in enumerate(plain):
        if posonly and i == posonly:
            parts.append("/")
        rendered = ast.unparse(arg)
        j = i - offset
        if 0 <= j < len(defaults):
            rendered = f"{rendered}={ast.unparse(defaults[j])}"
        parts.append(rendered)
    if args.vararg:
        parts.append("*" + ast.unparse(args.vararg))
    elif args.kwonlyargs:
        parts.append("*")
    for i, arg in enumerate(args.kwonlyargs):
        rendered = ast.unparse(arg)
        default = args.kw_defaults[i] if i < len(args.kw_defaults) else None
        if default is not None:
            rendered = f"{rendered}={ast.unparse(default)}"
        parts.append(rendered)
    if args.kwarg:
        parts.append("**" + ast.unparse(args.kwarg))
    return ", ".join(parts)


def _docstring_first_line(node: ast.AST) -> str | None:
    body = getattr(node, "body", None)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        doc = body[0].value.value
        if isinstance(doc, str) and doc.strip():
            return doc.strip().splitlines()[0][:100]
    return None


def _python_structural_summary_lines(source: str) -> tuple[list[str], int, int] | None:
    """``(symbol_lines, total_lines, elided_symbols)`` for a Python module.

    Declarations are kept; bodies are elided; every symbol carries its line
    range so the caller can teach the model exactly which range re-reads the
    body it just lost. Returns None when the file does not parse (the raw
    body is the honest answer for a broken file).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    out: list[str] = []

    def emit(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, depth: int) -> int:
        """Emit one symbol block; returns how many symbol lines it used."""
        used = 0
        pad = "    " * depth
        start, end = node.lineno, node.end_lineno or node.lineno
        for dec in getattr(node, "decorator_list", []):
            name = ast.unparse(dec)
            out.append(f"{pad}@{name}"[:110])
            used += 1
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            keyword = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
            out.append(f"{pad}{keyword} {node.name}({_format_args(node.args)}){returns}"[:110])
        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            head = f"class {node.name}" + (f"({bases})" if bases else "")
            out.append(f"{pad}{head}:"[:110])
        used += 1
        doc = _docstring_first_line(node)
        if doc:
            out.append(f'{pad}    "{doc}')
            used += 1
        out[-1] = f"{out[-1]}  ·  L{start}-{end}"
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    used += emit(child, depth + 1)
        return used

    module_doc = _docstring_first_line(tree)
    if module_doc:
        out.append(f'"{module_doc}')
    imports = sum(1 for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)))
    if imports:
        out.append(f"[imports: {imports} (elided)]")
    symbol_count = 0
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbol_count += emit(node, 0)
    if symbol_count == 0 and not module_doc and not imports:
        return None
    total_lines = len(source.splitlines())
    return out, total_lines, symbol_count


def _structural_summary_result(
    tool_call_id: str, path: Path, source: str, context: ToolContext | None
) -> ToolResult | None:
    """The read result for a Python file summarized instead of dumped, or
    None when the file should fall back to the raw body (parse failure,
    nothing declarative in it)."""
    parsed = _python_structural_summary_lines(source)
    if parsed is None:
        return None
    symbol_lines, total_lines, symbol_count = parsed
    if len(symbol_lines) > PYTHON_SUMMARY_MAX_SYMBOLS:
        omitted = len(symbol_lines) - PYTHON_SUMMARY_MAX_SYMBOLS
        symbol_lines = symbol_lines[:PYTHON_SUMMARY_MAX_SYMBOLS] + [
            f"[…{omitted} more symbol lines elided; narrow with grep, then range-read]"
        ]
    full = "\n".join(symbol_lines)
    # Fit the shown portion inside the char budget line-wholesale — a slice
    # mid-line would garble the very ranges the footer teaches the model to
    # use. Whatever does not fit goes to the spill like any other long list.
    shown_lines: list[str] = []
    budget = TOOL_OUTPUT_LIMIT_CHARS
    for line in symbol_lines:
        if budget - (len(line) + 1) < 0:
            break
        shown_lines.append(line)
        budget -= len(line) + 1
    shown = "\n".join(shown_lines)
    body, spill_details = _capped_list_body(full, shown, "read", context)
    header = f"{path} — structural summary ({total_lines} lines, " f"{symbol_count} symbol lines)"
    footer = (
        "\n[declaration-only view; bodies elided. Re-read exact code with "
        "range='start-end' (e.g. range='40-52'), or raw=true for the full text]"
    )
    details: dict[str, Any] = {"path": str(path), "summary": True}
    if spill_details:
        details.update(spill_details)
    return _text(tool_call_id, "read", header + "\n" + body + footer, details=details)


def _list_dir_entries(path: Path) -> list[str]:
    """One directory's entries, directories marked with a trailing ``/``.

    Synchronous by design: ``asyncio.to_thread`` is the only caller, and the
    shape is exactly what the loop-bound listing used to build inline.
    """
    return sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())


def _read_file_snapshot(path: Path) -> tuple[int, ImageInfo | None, bytes | None]:
    """Stat, classify and read one snapshot under the mutation stripe.

    ``None`` data means the classified snapshot exceeded its applicable cap.
    Keeping all three operations in one transaction prevents a writer from
    swapping text for an image (or a small file for an oversized one) between
    the limit/classification checks and the returned bytes.
    """
    with _file_transaction(path):
        size = path.stat().st_size
        info = sniff_image_file(str(path))
        limit = READ_IMAGE_LIMIT_BYTES if info else READ_FILE_LIMIT_BYTES
        data = None if size > limit else path.read_bytes()
        return size, info, data


def _decode_text_lines(data: bytes) -> tuple[str, list[str]]:
    """Decode file bytes as UTF-8 and split, preserving the source too.

    The source feeds Python structural summaries; rebuilding it from
    ``splitlines`` would normalize line endings and lose the exact text the
    parser and summary footer describe. Strict-decode-then-replace is
    preserved verbatim: only invalid UTF-8 pays the second pass.
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
    return text, text.splitlines()


@_guard("read")
async def execute_read(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Read a file (with optional line range) or resolve an internal URL."""
    try:
        params = ReadParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "read", exc)
    target = params.path.strip()
    if not target:
        return _error(tool_call_id, "read", "path must be a non-empty string")

    # Spill handles are served by this module, not the host resolver: the
    # resolver contract returns one whole string, and the whole value of a
    # handle is that it answers a RANGE. Checked first so a host that also
    # claims 'spill://' cannot shadow the expansion path a footer promised.
    if target.startswith(SPILL_SCHEME):
        return _read_spill(tool_call_id, target, params.range)

    # ``read https://…`` is sugar for a web fetch: it delegates to the SAME
    # engine web_fetch uses, returning the same bounded preview + spill handle,
    # so a URL becomes just another internal-ish URL that ``read`` resolves. Kept
    # BEFORE the internal-URL branch (which excludes http/https) and after the
    # spill branch, so it cannot shadow ``read spill://`` / ``read skill://`` /
    # ``read <file>`` — the regression guard those three keep working is exactly
    # what this ordering protects. Imported lazily to avoid a module cycle
    # (web_fetch.tool imports helpers from this module).
    if target.startswith(("http://", "https://")):
        from local_operator.web_fetch.tool import run_fetch

        preview, details, is_error = await run_fetch(
            target, tool_name="read", context=context, signal=signal
        )
        # A non-2xx fetch comes back is_error=True (F1). Carry ``details`` on the
        # error result too — unlike a plain read error, a fetch error still has a
        # final URL/status/http_error flag the card renders in its error
        # treatment, so the error path must not drop the structured payload the
        # success path keeps.
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="read",
            content=[TextContent(text=preview)],
            details=details,
            is_error=is_error,
        )

    # Internal URLs (skill://...) go through the session-installed resolver.
    if "://" in target and not target.startswith(("http://", "https://", "file://")):
        resolver = getattr(context, "resolve_internal_url", None) if context else None
        if resolver is None:
            return _error(
                tool_call_id,
                "read",
                f"Cannot resolve '{target}': no internal URL resolver is available.",
            )
        content = resolver(target)
        if content is None:
            return _error(
                tool_call_id,
                "read",
                f"Cannot resolve '{target}': the resolver does not handle this URL.",
            )
        # Deliberately NO supersede_key here. This path serves internal URLs
        # (skill://, guide://, mcp://), and skill reads are exempt from pruning
        # anyway -- ``_is_prunable`` protects them because a pruned skill just
        # gets re-read in a loop. Declaring a key would be inert for those and
        # would add avoidable risk for the rest, since a resolver result is not
        # always the same content under the same URL.
        return _text(tool_call_id, "read", content, details={"url": target})

    cwd = _safe_cwd(context)
    path, inside, resolvable = _resolve_workspace_path(target, cwd)
    if not path.exists():
        message = f"Path does not exist: {path}"
        # Skills are virtual resources, not files on disk. Point the agent to
        # skill://<name> when it tries to read a discovered or guessed SKILL.md.
        folded_parts = [p.casefold() for p in path.parts]
        if "skills" in folded_parts and path.name.casefold() == "skill.md":
            skill_name = path.parent.name
            resource = (
                f"skill://{skill_name}"
                if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", skill_name)
                else "skill://<name>"
            )
            message += (
                f". Skills are virtual resources loaded via `skill://`: read with `{resource}`."
            )
        return _error(tool_call_id, "read", message)

    # Outside-workspace reads escalate to an approval prompt regardless of
    # the read tier auto-approval the host normally applies.
    if not inside:
        description = _approval_description(path, inside, "read", resolvable)
        if not await _check_approval(context, "read", description):
            return _error(tool_call_id, "read", "User declined to read this file.")

    if path.is_dir():
        # iterdir in a thread: a wide directory (a node_modules, a build
        # output) is tens of thousands of stat calls, and the loop this
        # coroutine rides is the one rendering the TUI.
        entries = await asyncio.to_thread(_list_dir_entries, path)
        return _text(
            tool_call_id,
            "read",
            f"Directory listing of {path} ({len(entries)} entries):\n" + "\n".join(entries),
            details={"path": str(path)},
        )

    # Stat, content sniff and body read are one worker-thread transaction.
    # Classification is by CONTENT, never extension — and it must describe
    # the same bytes returned below. A concurrent in-process edit/write takes
    # the same path stripe and cannot swap the file between these decisions.
    size, info, data = await asyncio.to_thread(_read_file_snapshot, path)
    limit = READ_IMAGE_LIMIT_BYTES if info else READ_FILE_LIMIT_BYTES
    if data is None:
        advice = (
            "Resize it first (bash + sips/magick)."
            if info
            else "Use bash (head/tail) or a 'range' on a smaller file."
        )
        return _error(
            tool_call_id,
            "read",
            f"File too large to read ({size} bytes; limit {limit} bytes): {path}. {advice}",
        )

    if info:
        try:
            payload, wire_mime, summary = await asyncio.to_thread(bound_image_for_model, data, info)
        except ValueError as exc:
            # A text error, never an image block. Forwarding undecodable bytes
            # gets a 400 from the provider that no retry clears, because the
            # bad block is already in the transcript.
            return _error(tool_call_id, "read", f"Cannot read {path} as an image: {exc}")
        caption = f"Image {path} ({summary})"
        if params.range:
            # Silently dropping it would leave the model believing it read a
            # slice of something.
            caption += " — 'range' does not apply to an image and was ignored"
        return _image(
            tool_call_id,
            "read",
            caption,
            payload,
            wire_mime,
            details={"path": str(path), "mime_type": wire_mime},
        )

    if b"\x00" in data[:8000]:
        guessed = mimetypes.guess_type(path.name)[0] or ""
        if guessed.startswith("image/"):
            # The extension is the only evidence left, and it says image. Name
            # the format instead of reporting a generic binary: "not readable
            # as text" reads as a bug in read when the caller can see a .bmp.
            return _error(
                tool_call_id,
                "read",
                f"Unsupported image format ({guessed}): {path}. read returns PNG, JPEG, "
                "GIF, WebP and HEIC; convert it first (bash + sips/magick).",
            )
        return _error(tool_call_id, "read", f"Binary file not readable as text: {path}")

    # Decode + split in a thread: a 2 MB text read is a 2 MB decode and a
    # full pass to break lines, and the same loop renders the TUI. Keep the
    # source beside the lines for the structural-summary path below.
    text, lines = await asyncio.to_thread(_decode_text_lines, data)

    if params.range:
        try:
            start, end = _parse_line_range(params.range)
        except ValueError as exc:
            return _error(tool_call_id, "read", str(exc))
        selected = lines[start - 1 : end]
        if not selected:
            return _text(
                tool_call_id,
                "read",
                f"(range {params.range} is beyond end of file {path})",
                useless=True,
                details={"path": str(path), "useless": True},
            )
        return _text(
            tool_call_id,
            "read",
            _clamp_file_body(_number_lines(selected, start), path, start, len(lines)),
            # The range rides in details: compaction's supersede key must
            # distinguish ranged reads of the same file, or a read of lines
            # 900-1000 blanks an unrelated 1-100 read as "superseded".
            details={"path": str(path), "range": params.range},
        )
    # Python files read whole get a declaration-only structural summary:
    # the model sees the symbol table (with line ranges) instead of the full
    # body, and re-reads only the ranges it needs. The repo's own benchmark
    # put tool results at 87-96% of context bytes — the read tool is where
    # those bytes come from. Opt-outs: a range (explicit interest in exact
    # lines) or raw=true; unparseable files fall back to the body honestly.
    if (
        not params.range
        and not params.raw
        and path.suffix == ".py"
        and len(lines) >= PYTHON_SUMMARY_MIN_LINES
    ):
        # ``ast.parse`` plus the declaration walk are pure-Python CPU over
        # the whole file. Main added this path after the original liveness
        # fix; it belongs behind the same thread boundary as decode, or a
        # large Python read simply reintroduces the render-loop stall under
        # a new name.
        summarized = await asyncio.to_thread(
            _structural_summary_result, tool_call_id, path, text, context
        )
        if summarized is not None:
            return summarized

    if len(lines) > READ_LINE_CAP:
        body = _number_lines(lines[:READ_LINE_CAP], 1)
        remaining = len(lines) - READ_LINE_CAP
        return _text(
            tool_call_id,
            "read",
            _clamp_file_body(body, path, 1, len(lines))
            + f"\n\n[{remaining} more lines in file. Use range to continue]",
            details={"path": str(path)},
        )
    return _text(
        tool_call_id,
        "read",
        _clamp_file_body(_number_lines(lines, 1), path, 1, len(lines)) if lines else "(empty file)",
        details={"path": str(path)},
    )


def build_read_tool() -> AgentTool:
    return AgentTool(
        name="read",
        label="Read",
        description=(
            "Read a file, line range, or internal URL (skill://, guide://, mcp://). "
            "PNG/JPEG/GIF/WebP/HEIC files come back as a viewable image. "
            "Python files read whole return a structural summary; use a "
            "range or raw=true for exact text."
        ),
        parameters=ReadParams.model_json_schema(),
        approval_tier="read",
        # read model: parallel reads are the common batch shape.
        concurrency="shared",
        interruptible=False,
        execute=execute_read,
    )


# ---------------------------------------------------------------------------
# edit
# ---------------------------------------------------------------------------


class EditHunk(BaseModel):
    """One SEARCH/REPLACE hunk. ``old_text`` is matched exactly first, then
    with a whitespace-tolerant line matcher, so an edit written from a
    structural summary or from memory does not fail on indentation drift."""

    model_config = ConfigDict(extra="forbid")

    old_text: str = Field(
        description="Text to find (exact match tried first, then whitespace-tolerant)."
    )
    new_text: str = Field(
        description="Replacement text (re-indented to the file's level on the tolerant path)."
    )
    replace_all: bool = Field(
        default=False,
        description="Replace every occurrence of this hunk instead of requiring exactly one.",
    )


class EditParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="File to edit.")
    edits: list[EditHunk] = Field(
        default_factory=list,
        description=(
            "Hunks applied in order — the multi-edit form. Prefer this for "
            "several changes to one file: it costs one call instead of N."
        ),
    )
    old_text: str | None = Field(
        default=None,
        description="Single-edit form: exact text to find. Mutually exclusive with 'edits'.",
    )
    new_text: str | None = Field(default=None, description="Single-edit form: replacement text.")
    replace_all: bool = Field(
        default=False,
        description="Single-edit form: replace every occurrence instead of requiring exactly one.",
    )
    anchor_line: int | None = Field(
        default=None,
        description=(
            "Optional 1-based line that disambiguates a match that occurs in "
            "several places: the window containing this line wins."
        ),
    )

    @model_validator(mode="after")
    def _single_form_is_complete(self) -> "EditParams":
        """One form or the other, whole — a lone old_text is a modeling
        mistake pydantic reports in the tool's clean 'invalid arguments'
        voice rather than a bespoke message the caller cannot predict."""
        if (self.old_text is None) != (self.new_text is None):
            raise ValueError("single-edit form needs both old_text and new_text")
        if self.old_text is not None and self.edits:
            raise ValueError("pass either 'edits' (list of hunks) or old_text/new_text, not both")
        if not self.edits and self.old_text is not None and not self.old_text:
            raise ValueError("old_text must be a non-empty string")
        return self


def _leading_ws(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _match_windows(content: str, old_text: str) -> list[tuple[int, int, str]]:
    """``(start, end, matched_text)`` windows where ``old_text`` occurs.

    Pass 1 is exact substring matching — the anchored, honest form. Pass 2
    runs only when pass 1 found nothing: a line-based tolerant matcher that
    compares lines with ``strip()`` equality and returns the matched FILE
    text so the replacement can be re-indented per line into place.
    Tolerance is one-directional by design: an exact match is never silently
    flexed, and a fuzzy match never silently wins over an exact one that
    exists elsewhere.
    """
    windows: list[tuple[int, int, str]] = []
    start = content.find(old_text)
    while start != -1:
        windows.append((start, start + len(old_text), old_text))
        start = content.find(old_text, start + 1)
    if windows:
        return windows

    file_lines = content.splitlines(keepends=True)
    # Per-line start offsets so a window can be returned as a char span.
    offsets: list[int] = []
    at = 0
    for line in file_lines:
        offsets.append(at)
        at += len(line)
    old_lines = old_text.splitlines()
    if not old_lines or len(old_lines) > len(file_lines):
        return []

    def _tolerant(file_line: str, old_line: str) -> bool:
        return file_line.strip() == old_line.strip()

    for i in range(len(file_lines) - len(old_lines) + 1):
        window = file_lines[i : i + len(old_lines)]
        if all(_tolerant(f.rstrip("\r\n"), o) for f, o in zip(window, old_lines)):
            # The span runs to the end of the last matched line INCLUDING its
            # newline, so replacing it cannot splice the following line onto
            # the replacement's final line.
            last_line = window[-1]
            end = offsets[i + len(old_lines) - 1] + len(last_line)
            matched = "".join(window)
            windows.append((offsets[i], end, matched))
    return windows


def _reindent_new_text(new_text: str, matched: str, old_text: str) -> str:
    """Re-anchor a TOLERANT match while retaining the file's line endings.

    Exact matches bypass this function and insert ``new_text`` byte-for-byte.
    On the tolerant path, indentation follows the file window and every model
    newline adopts the corresponding file line's CRLF/LF spelling. Extra
    inserted lines borrow the final matched line's indentation and EOL.
    """
    file_lines = matched.splitlines(keepends=True)
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    if not old_lines or not file_lines:
        return new_text

    def split_ending(line: str) -> tuple[str, str]:
        body = line.rstrip("\r\n")
        return body, line[len(body) :]

    out: list[str] = []
    for i, line in enumerate(new_lines):
        body, model_ending = split_ending(line)
        if not body.strip():
            ref = min(i, len(file_lines) - 1)
            _file_body, file_ending = split_ending(file_lines[ref])
            out.append(file_ending if model_ending else "")
            continue
        ref = min(i, len(file_lines) - 1, len(old_lines) - 1)
        file_body, file_ending = split_ending(file_lines[ref])
        old_body, _old_ending = split_ending(old_lines[ref])
        file_indent = _leading_ws(file_body)
        old_indent = _leading_ws(old_body)
        line_indent = _leading_ws(body)
        if line_indent.startswith(old_indent):
            indentation = file_indent + line_indent[len(old_indent) :]
        else:
            # Mixed tab/space model indentation with no literal prefix: keep
            # the file's anchor and conservatively express only the extra
            # relative depth as spaces.
            indentation = file_indent + " " * max(len(line_indent) - len(old_indent), 0)
        ending = (file_ending or model_ending) if model_ending else ""
        out.append(indentation + body.lstrip(" \t") + ending)
    return "".join(out)


def _line_of_offset(content: str, offset: int) -> int:
    return content.count("\n", 0, offset) + 1


@_guard("edit")
async def execute_edit(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Ordered SEARCH/REPLACE hunks in a file, one call for the whole change.

    Why hunks and not whole-file writes: a ``write`` argument IS the file
    body, billed as model output (the most expensive token class, and never
    cacheable) and then re-billed as prompt on every later turn. On build-heavy
    work that dominates the bill — the repo's own benchmark measured file
    bodies at 48-74% of task cost. The hunk form prices an edit by what
    CHANGED.

    Ambiguity is an error, not a guess: a hunk matching more than one place
    without ``replace_all`` (or a disambiguating ``anchor_line``) refuses,
    because silently editing the first occurrence is how edits corrupt the
    wrong site.
    """
    try:
        params = EditParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "edit", exc)
    if not params.path.strip():
        return _error(tool_call_id, "edit", "path must be a non-empty string")

    hunks: list[EditHunk] = list(params.edits)
    if params.old_text is not None and params.new_text is not None:
        hunks.append(
            EditHunk(
                old_text=params.old_text, new_text=params.new_text, replace_all=params.replace_all
            )
        )
    if not hunks:
        return _error(
            tool_call_id,
            "edit",
            "nothing to edit: pass 'edits' (preferred) or old_text/new_text.",
        )

    path, inside, _resolvable = _resolve_workspace_path(params.path, _safe_cwd(context))
    if not path.is_file():
        return _error(tool_call_id, "edit", f"File does not exist: {path}")

    # Read/match/replace/write/diff in a thread. Main's multi-hunk edit grew
    # substantially after the first liveness fix, but its contract is the
    # same load: file IO plus pure-Python matching and difflib over the whole
    # file. Keeping that work on the Textual loop merely moved the reported
    # freeze from the old one-hunk path into the new implementation.
    outcome = await asyncio.to_thread(_edit_file_result, path, hunks, params.anchor_line)
    if isinstance(outcome, str):
        return _error(tool_call_id, "edit", outcome)
    total_replacements, details = outcome
    return _text(
        tool_call_id,
        "edit",
        f"Edited {path}: {len(hunks)} hunk(s), {total_replacements} replacement(s) applied.",
        details=details,
    )


def _edit_file_result(
    path: Path,
    hunks: list[EditHunk],
    anchor_line: int | None,
) -> tuple[int, dict[str, Any]] | str:
    """Serialize one file transaction across parent/child AgentLoops."""
    with _file_transaction(path):
        return _edit_file_result_locked(path, hunks, anchor_line)


def _edit_file_result_locked(
    path: Path,
    hunks: list[EditHunk],
    anchor_line: int | None,
) -> tuple[int, dict[str, Any]] | str:
    """The current multi-hunk edit engine, synchronous for ``to_thread``.

    A string is the exact refusal the tool returns. Counting, ambiguity
    resolution and mutation all happen against one file snapshot, so moving
    the engine off-loop cannot split the read/decide/write transaction.
    """
    with path.open("r", encoding="utf-8", newline="") as stream:
        original = stream.read()
    current = original
    total_replacements = 0

    for index, hunk in enumerate(hunks):
        if hunk.old_text == "":
            return f"hunk {index + 1}: old_text must be non-empty"
        windows = _match_windows(current, hunk.old_text)
        if not windows:
            advice = (
                f" — or the range around line {anchor_line}"
                if anchor_line
                else " to get the current text"
            )
            return (
                f"hunk {index + 1}: old_text not found (exact and whitespace-tolerant "
                f"matchers both failed). Re-read the file{advice} and retry."
            )
        chosen = windows
        if len(windows) > 1 and not hunk.replace_all:
            if anchor_line is not None:
                anchored = []
                for window in windows:
                    first_line = _line_of_offset(current, window[0])
                    last_line = _line_of_offset(current, max(window[1] - 1, window[0]))
                    if first_line <= anchor_line <= last_line:
                        anchored.append(window)
                if len(anchored) == 1:
                    chosen = anchored
            if len(chosen) > 1:
                return (
                    f"hunk {index + 1}: old_text matches {len(chosen)} places; include "
                    "more surrounding context, give anchor_line, or set replace_all=true."
                )
        # Apply back-to-front so earlier offsets stay valid within this hunk.
        for start, end, matched in sorted(chosen, key=lambda window: window[0], reverse=True):
            exact = matched == hunk.old_text
            replacement = (
                hunk.new_text
                if exact
                else _reindent_new_text(hunk.new_text, matched, hunk.old_text)
            )
            if (
                not exact
                and matched.endswith(("\n", "\r"))
                and not replacement.endswith(("\n", "\r"))
            ):
                replacement += "\r\n" if matched.endswith("\r\n") else "\n"
            current = current[:start] + replacement + current[end:]
            total_replacements += 1

    if current != original:
        with path.open("w", encoding="utf-8", newline="") as stream:
            stream.write(current)
    return total_replacements, _diff_details(str(path), original, current)


def build_edit_tool() -> AgentTool:
    return AgentTool(
        name="edit",
        label="Edit",
        describe_approval=_describe_path_approval("edit"),
        description=(
            "Apply ordered SEARCH/REPLACE hunks to a file ('edits' list for "
            "several changes in one call; exact match first, then "
            "whitespace-tolerant; anchor_line disambiguates repeats)."
        ),
        parameters=EditParams.model_json_schema(),
        approval_tier="write",
        # edit model: two concurrent edits on one file corrupt each
        # other's match anchors; exclusive serializes the read-modify-write.
        concurrency="exclusive",
        interruptible=False,
        execute=execute_edit,
        resource_keys=_file_resource_keys,
    )


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------


class WriteParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="File path to create or overwrite.")
    content: str = Field(description="Full file content to write.")


def _line_delta(before: str, after: str) -> tuple[int, int]:
    """Line counts added/removed between two file states, for the UI's +N/-N.

    Uses a real sequence match rather than a length difference so a same-size
    rewrite still reports its churn, and a pure insertion does not falsely
    report removals. Whole-file replacement (write over an existing file) and
    a one-hunk edit both funnel through here so the two tools cannot drift.

    Cheap by construction: SequenceMatcher over LINES (not characters), and
    the inputs are files a human is editing, so this is not a hot path.
    """
    if before == after:
        return 0, 0
    old_lines = before.splitlines()
    new_lines = after.splitlines()
    if not before:
        return len(new_lines), 0
    if not after:
        return 0, len(old_lines)
    added = removed = 0
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            removed += i2 - i1
            added += j2 - j1
        elif tag == "delete":
            removed += i2 - i1
        elif tag == "insert":
            added += j2 - j1
    return added, removed


#: The unified-diff payload cap. The diff rides the tool result's ``details``,
#: which the transcript PERSISTS — a runaway file write must not let one tool
#: result grow the transcript without bound. A typical edit is a handful of
#: hunks; this admits a large one while keeping the stored payload sane. The
#: TUI's expanded card has its own display cap; the stored cap is about the
#: ledger, not the screen.
_DIFF_DETAILS_CAP_LINES = 200


def _diff_details(path: str, before: str, after: str) -> dict[str, Any]:
    """The write/edit tool-result details: line counts + a rendered unified diff.

    Both write and edit funnel their before/after states through a real
    sequence match, so the UI's ``+N/-N`` counters and the expanded card's
    diff view describe the SAME change and can never disagree about what
    happened. The diff itself powers the TUI's expanded view; it is capped
    (see ``_DIFF_DETAILS_CAP_LINES``) so the persisted details stay bounded.
    """
    added, removed = _line_delta(before, after)
    if not added and not removed:
        return {"path": str(path), "added": 0, "removed": 0}
    diff = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            lineterm="",
            n=2,
        )
    )
    if len(diff) > _DIFF_DETAILS_CAP_LINES:
        diff = diff[:_DIFF_DETAILS_CAP_LINES] + ["…"]
    return {"path": str(path), "added": added, "removed": removed, "diff": diff}


@_guard("write")
async def execute_write(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Create or overwrite a file, creating parent directories as needed."""
    try:
        params = WriteParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "write", exc)
    if not params.path.strip():
        return _error(tool_call_id, "write", "path must be a non-empty string")
    # Write-tier approval is the loop's gate; see execute_bash.
    path, inside, _resolvable = _resolve_workspace_path(params.path, _safe_cwd(context))

    # The read-modify-write-diff block runs in a thread: the loop this
    # coroutine rides is the SAME loop that renders the TUI, and a previous
    # revision of a rewritten file plus the unified diff over it are
    # megabyte-scale CPU (difflib is pure Python) at exactly the moment a
    # concurrent sibling tool is also settling. On the loop that read as the
    # intermittent whole-screen freeze; off it, the frame keeps animating.
    existed, details = await asyncio.to_thread(_write_file_result, path, params.content)
    verb = "Overwrote" if existed else "Created"
    return _text(
        tool_call_id,
        "write",
        f"{verb} {path} ({len(params.content)} chars).",
        details=details,
    )


def _write_file_result(path: Path, content: str) -> tuple[bool, dict[str, Any]]:
    """Serialize one overwrite transaction across parent/child AgentLoops."""
    with _file_transaction(path):
        return _write_file_result_locked(path, content)


def _write_file_result_locked(path: Path, content: str) -> tuple[bool, dict[str, Any]]:
    """The filesystem half of ``write``: read prior, write new, diff both.

    Synchronous by design — ``asyncio.to_thread`` is the only caller, and
    keeping it a plain function means the abort/approval decisions stay on
    the loop where their timeouts live.
    """
    existed = path.exists()
    previous = ""
    if existed:
        try:
            previous = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            # Binary or unreadable prior content: we still write, we just
            # cannot report a meaningful diff for it.
            previous = ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return existed, _diff_details(str(path), previous, content)


def build_write_tool() -> AgentTool:
    return AgentTool(
        name="write",
        label="Write",
        describe_approval=_describe_path_approval("write"),
        description="Create or overwrite a file (parents are created automatically).",
        parameters=WriteParams.model_json_schema(),
        approval_tier="write",
        # write model: concurrent writes to the same file race silently;
        # an exclusive tool makes the last-writer outcome deterministic.
        concurrency="exclusive",
        interruptible=False,
        execute=execute_write,
        resource_keys=_file_resource_keys,
    )


# ---------------------------------------------------------------------------
# gitignore-aware walking (shared by glob and grep)
# ---------------------------------------------------------------------------


class _IgnoreRule:
    """One compiled .gitignore/.ignore line, scoped to its file's directory.

    Semantics follow gitignore(5) closely enough for search pruning: ``*``
    does not cross ``/``, ``**`` does, a trailing ``/`` matches directories
    only, a leading or interior ``/`` anchors the pattern to the rule's base
    directory, otherwise the pattern matches a basename at any depth below
    it, and a later ``!`` rule re-includes. Why this exists at all: the old
    fixed prune list covered .git/node_modules & co but never the project's
    OWN ignored paths — build dirs, vendored trees, virtualenvs under other
    names — so search results carried exactly the noise the project had
    already declared dead.
    """

    __slots__ = ("regex", "negated", "dir_only", "base")

    def __init__(self, pattern: str, base: str) -> None:
        self.negated = pattern.startswith("!")
        if self.negated:
            pattern = pattern[1:]
        self.dir_only = pattern.endswith("/")
        pattern = pattern.rstrip("/")
        self.base = base
        anchored = pattern.startswith("/") or "/" in pattern
        pattern = pattern.lstrip("/")
        parts = pattern.split("/")
        regex = "/".join(".*" if seg == "**" else _glob_segment_regex(seg) for seg in parts)
        if self.dir_only:
            # A directory pattern owns its whole subtree, not just the
            # directory entry: "dist/" must match "dist/out.js" too, or a
            # post-filtering caller (glob) would admit every file under an
            # ignored directory the walker (grep) would have pruned.
            regex += "(/.*)?"
        if anchored:
            prefix = re.escape(base + "/") if base else ""
            self.regex = re.compile("^" + prefix + regex)
        else:
            self.regex = re.compile("(^|.*/)" + regex + "$")


def _glob_segment_regex(segment: str) -> str:
    """One path segment of a gitignore pattern as regex (``*`` stays in-segment)."""
    out: list[str] = []
    i = 0
    while i < len(segment):
        ch = segment[i]
        if ch == "*":
            if segment[i : i + 2] == "**":
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
            i += 1
            continue
        if ch == "?":
            out.append("[^/]")
            i += 1
            continue
        out.append(re.escape(ch))
        i += 1
    return "".join(out)


def _load_ignore_rules(directory: Path, rel_dir: str) -> list[_IgnoreRule]:
    """Rules from a directory's ``.gitignore``/``.ignore``, empty when absent."""
    rules: list[_IgnoreRule] = []
    for name in (".gitignore", ".ignore"):
        file = directory / name
        if not file.is_file():
            continue
        try:
            raw = file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in raw.splitlines():
            line = line.rstrip("\r")
            if not line.strip() or line.startswith("#"):
                continue
            rules.append(_IgnoreRule(line, rel_dir))
    return rules


def _ignored(
    rel: str,
    is_dir: bool,
    rules: list[tuple[str, list[_IgnoreRule]]],
) -> bool:
    """gitignore last-match-wins evaluation over the ancestor rule stack."""
    ignored = False
    for _base, base_rules in rules:
        for rule in base_rules:
            if rule.regex.search(rel):
                ignored = not rule.negated
    return ignored


def _walk_entries(root: Path, *, respect_ignore: bool = True) -> list[Path]:
    """Depth-first walk pruning VCS/vendor/build trees, dotdirs, symlinks and
    (when ``respect_ignore``) gitignore-declared paths. Files only.

    Git semantics for directories are honored structurally: an ignored
    directory's subtree is skipped outright, so a ``!`` rule cannot re-include
    inside it — exactly git's own behaviour.
    """
    files: list[Path] = []

    def _walk(directory: Path, rel_dir: str, rules: list[tuple[str, list[_IgnoreRule]]]) -> None:
        local_rules = rules
        if respect_ignore:
            found = _load_ignore_rules(directory, rel_dir)
            if found:
                local_rules = rules + [(rel_dir, found)]
        # os.scandir, not Path.iterdir + per-entry Path.is_symlink/is_dir/
        # is_file: a DirEntry carries the d_type the kernel already returned
        # with the directory listing, so the symlink/dir/file classification
        # below costs zero extra syscalls on the common path where iterdir
        # paid three stat(2) calls PER ENTRY. On the 60k-entry trees this
        # walk actually meets (a workspace with vendored checkouts) that is
        # the difference between a scan and a stall — and this function runs
        # under a worker thread on behalf of a TUI whose frame budget the
        # caller is protecting, so raw walk speed is part of the contract.
        # Sorting by name matches the old sorted(iterdir()) order exactly:
        # within one directory, Path ordering IS name ordering.
        try:
            with os.scandir(directory) as scan:
                entries = sorted(scan, key=lambda e: e.name)
        except OSError:
            return
        for entry in entries:
            try:
                if entry.is_symlink():
                    continue  # never follow links: cycles and out-of-tree escapes
                is_dir = entry.is_dir(follow_symlinks=False)
                is_file = entry.is_file(follow_symlinks=False)
            except OSError:
                # An entry that vanished or cannot be classified mid-walk is
                # skipped, matching the directory-level OSError handling: a
                # concurrent delete is normal filesystem life, not an error.
                continue
            rel = f"{rel_dir}/{entry.name}" if rel_dir else entry.name
            if is_dir:
                if entry.name in _GREP_PRUNE_DIRS or entry.name.startswith("."):
                    continue
                if respect_ignore and _ignored(rel, True, local_rules):
                    continue
                _walk(Path(entry.path), rel, local_rules)
            elif is_file:
                if respect_ignore and _ignored(rel, False, local_rules):
                    continue
                files.append(Path(entry.path))

    _walk(root, "", [])
    return files


def _walk_files(root: Path) -> list[Path]:
    """The grep file set: the ignore-aware walk."""
    return _walk_entries(root)


def _grep_file_set(target: Path) -> tuple[list[Path], Path]:
    """``(files, base)`` for one grep target, file or directory.

    Synchronous by design: the walk is tens of thousands of scandir/stat
    calls on a large tree, so every caller reaches this through
    ``asyncio.to_thread``. Running it inline in ``execute_grep`` was the
    freeze reported as "the session hangs on concurrent tool calls": under
    Textual's eager task factory the runner coroutine executes synchronously
    up to its first true suspension AT TASK-CREATION TIME, so a batch of
    greps put every tree walk back-to-back on the render loop before the
    first frame of the batch could paint (sampled live: 100% of main-thread
    samples inside os_lstat/os_scandir/os_stat under task_eager_start).
    """
    if target.is_file():
        return [target], target.parent
    return _walk_files(target), target


def _count_oversized_files(target: Path) -> int:
    """How many files in the grep set exceed ``GREP_FILE_LIMIT_BYTES``.

    The ripgrep engine applies ``--max-filesize`` silently, and the footer
    contract promises the skipped count either way — this recovers it. It
    re-walks the tree, which is exactly the filesystem load described on
    ``_grep_file_set``, so it is synchronous and thread-hosted for the same
    reason.
    """
    files, _base = _grep_file_set(target)
    skipped = 0
    for file_path in files:
        try:
            if file_path.stat().st_size > GREP_FILE_LIMIT_BYTES:
                skipped += 1
        except OSError:
            continue
    return skipped


class GlobParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: str = Field(
        description="Glob pattern relative to the working directory ('**/*.py' supported)."
    )


def _literal_prefix(pattern: str) -> str:
    """The leading run of the pattern with no glob metacharacters — the part
    that names directories the author EXPLICITLY asked to descend into."""
    out = []
    for ch in pattern:
        if ch in "*?[":
            break
        out.append(ch)
    return "".join(out).rstrip("/")


def _path_is_ignored(root: Path, path: Path) -> bool:
    """Evaluate root + nested ignore files for one glob candidate.

    Unlike grep's walker, pathlib.glob materializes candidates without walking
    through our rule stack. Rebuild the ancestor stack here so a
    packages/a/.gitignore has the same authority over `**/*.py` as it does in
    grep. The caller still bypasses this for an explicitly named literal
    prefix ("dist/*.js" means the ignored dist on purpose).
    """
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        return False
    rules: list[tuple[str, list[_IgnoreRule]]] = []
    current = root
    rel_dir = ""
    for index, part in enumerate(rel_parts):
        found = _load_ignore_rules(current, rel_dir)
        if found:
            rules.append((rel_dir, found))
        rel = "/".join(rel_parts[: index + 1])
        candidate = current / part
        if _ignored(rel, candidate.is_dir(), rules):
            return True
        current = candidate
        rel_dir = rel
    return False


def _glob_walk(root: Path, pattern: str) -> list[str]:
    """The walk half of execute_glob, run in a worker thread.

    Matching still uses pathlib (so explicit hidden/vendor components in the
    pattern work), then gitignore-declared paths are filtered out — unless
    the pattern's literal prefix names them, because an author who writes
    'dist/index.html' into a repo that ignores dist/ means that file."""
    prefix = _literal_prefix(pattern)
    out = []
    for p in root.glob(pattern):
        rel = p.relative_to(root).as_posix()
        explicitly_named = bool(prefix) and (rel == prefix or rel.startswith(prefix + "/"))
        if not explicitly_named and _path_is_ignored(root, p):
            continue
        out.append(rel + ("/" if p.is_dir() else ""))
    return sorted(out)


@_guard("glob")
async def execute_glob(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """List paths matching a glob pattern (files and directories)."""
    try:
        params = GlobParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "glob", exc)
    pattern = params.pattern.strip()
    if not pattern:
        return _error(tool_call_id, "glob", "pattern must be a non-empty string")
    if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
        message = (
            "pattern must be a relative glob within the working directory "
            "(no absolute paths, no '..')."
        )
        # Skills are virtual resources, not files the model should discover.
        # Correct only this observed misuse so every ordinary glob error keeps
        # its concise, generic diagnostic.
        if Path(pattern).is_absolute() and Path(pattern).name.casefold() == "skill.md":
            # A conservative resource segment avoids reflecting path syntax or
            # prose into a URL while still covering normal catalog names.
            skill_name = Path(pattern).parent.name
            resource = (
                f"skill://{skill_name}"
                if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", skill_name)
                else "skill://<name>"
            )
            message += (
                " Do not scan the filesystem for SKILL.md; read the selected skill "
                f"directly with `{resource}`."
            )
        return _error(tool_call_id, "glob", message)

    root = Path(_safe_cwd(context))
    # An unbounded ``**`` walk is filesystem work that can freeze the session;
    # off the event loop and raced against abort like the grep scan.
    matches, aborted = await _run_with_abort(
        asyncio.to_thread(_glob_walk, root, pattern),
        signal,
        lambda: None,
    )
    if aborted:
        return _error(tool_call_id, "glob", "Glob aborted.")
    if not matches:
        return _text(
            tool_call_id,
            "glob",
            f"No paths matched pattern '{params.pattern}'.",
            useless=True,
            details={"useless": True},
        )
    # Spill the COMPLETE list before capping. The 500-path cap silently threw
    # the tail away, so a model looking for a file that sorted late was told
    # it did not exist; now the whole list is one range read away. The join
    # and spill ride a thread because a whole-tree glob is thousands of
    # paths and the spill a disk write of all of them.
    total = len(matches)
    shown = matches[:GLOB_RESULT_LIMIT]
    body, spill_details = await asyncio.to_thread(
        _joined_capped_list_body, matches, shown, "glob", context
    )
    header = f"{len(shown)} match(es) for '{params.pattern}'"
    if total > len(shown):
        header += f" of {total} (capped at {GLOB_RESULT_LIMIT})"
    return _text(tool_call_id, "glob", header + ":\n" + body, details=spill_details)


def build_glob_tool() -> AgentTool:
    return AgentTool(
        name="glob",
        label="Glob",
        description=(
            "Find files and directories by glob pattern ('**' supported); "
            "gitignore-declared paths are excluded unless the pattern names them."
        ),
        parameters=GlobParams.model_json_schema(),
        approval_tier="read",
        # Read-only listing; parallel globs are the common batch shape.
        concurrency="shared",
        interruptible=False,
        execute=execute_glob,
    )


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------


class GrepParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: str = Field(description="Python regular expression to search for.")
    path: str = Field(
        default=".",
        description="Directory or file to search (relative to the working directory).",
    )
    include: str | None = Field(
        default=None,
        description=(
            "Optional glob filter applied to file names/basenames " "(e.g. '*.py', '**/*.ts')."
        ),
    )
    case: bool = Field(default=True, description="Case-sensitive matching.")
    context_lines: int = Field(
        default=0,
        ge=0,
        le=10,
        description=(
            "Lines of surrounding context per match (like grep -C). Context "
            "lines render as 'path:line-text' (dash) vs matches 'path:line:text'."
        ),
    )
    skip: int = Field(
        default=0,
        ge=0,
        description="Skip the first N matches (pagination for large result sets).",
    )


def _glob_matches(rel_path: str, pattern: str) -> bool:
    """Match ``rel_path`` against ``pattern`` (basename fallback for bare globs)."""
    if fnmatch.fnmatch(rel_path, pattern):
        return True
    name = rel_path.rsplit("/", 1)[-1]
    return fnmatch.fnmatch(name, pattern)


#: Wall-clock cap for one grep scan. Bounds the pathological-regex case
#: (backtracking patterns on large lines) without classifying regexes; a
#: scan that hits it returns what it has so far.
GREP_SCAN_DEADLINE_S = 30.0

#: How many matches the scan COLLECTS, versus the 200 it displays. The
#: displayed cap protects the prompt; this one protects the machine, and they
#: are different numbers because the spill sits between them — an agent that
#: greps a large tree gets 200 matches in context and the other 4,800 behind a
#: handle it can search. Stopping the scan at 200, as it used to, made the
#: handle a copy of what was already on screen and left "capped at 200" as a
#: dead end with no way to see match 201.
GREP_SPILL_MATCH_LIMIT = 5000


#: Engine selection: 'auto' uses ripgrep when the binary is on PATH (native
#: speed, native gitignore/hidden semantics) and falls back to the Python
#: scan otherwise; 'python' forces the fallback for deterministic tests.
#: Read per call, not cached at import, so a test (or a user shell) can pin
#: the engine without reimporting the module.
def _grep_engine() -> str:
    return os.environ.get("LOCAL_OPERATOR_GREP_ENGINE", "auto")


_RG_PATH: str | None = None
_RG_RESOLVED = False


def _rg_binary() -> str | None:
    """Absolute ripgrep path when usable, else None (cached)."""
    global _RG_PATH, _RG_RESOLVED
    if not _RG_RESOLVED:
        import shutil

        _RG_PATH = shutil.which("rg")
        _RG_RESOLVED = True
    return _RG_PATH


def _use_ripgrep() -> bool:
    return _grep_engine() == "auto" and _rg_binary() is not None


def _match_record(rel: str, lineno: int, line: str, kind: str) -> tuple[str, int, str, str]:
    return (rel, lineno, line, kind)


def _python_grep_scan(
    files: list[Path],
    base: Path,
    regex: re.Pattern[str],
    include: str | None,
    context_lines: int,
) -> tuple[list[tuple[str, int, str, str]], int, int]:
    """The pure-Python scan, run in a worker thread.

    Returns ``(records, files_searched, files_skipped)`` where a record is
    ``(rel, lineno, text, kind)`` and kind is ``'m'`` (match) or ``'c'``
    (context). Context records are only produced when ``context_lines`` > 0.
    Kept synchronous and self-contained so ``asyncio.to_thread`` can carry it
    off the event loop; the deadline bounds a backtracking pattern without
    touching the loop.
    """
    deadline = time.monotonic() + GREP_SCAN_DEADLINE_S
    records: list[tuple[str, int, str, str]] = []
    files_searched = 0
    files_skipped = 0
    for file_path in files:
        if time.monotonic() > deadline:
            break
        rel = (
            file_path.relative_to(base).as_posix()
            if base in file_path.parents or file_path == base
            else file_path.as_posix()
        )
        if include and not _glob_matches(rel, include):
            continue
        try:
            if file_path.stat().st_size > GREP_FILE_LIMIT_BYTES:
                files_skipped += 1
                continue
            data = file_path.read_bytes()
        except OSError:
            continue
        if b"\x00" in data[:8000]:
            continue  # binary file
        files_searched += 1
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()
        hit_lines = [n for n, line in enumerate(lines, start=1) if regex.search(line)]
        if not hit_lines:
            continue
        wanted: set[int] = set()
        for n in hit_lines:
            wanted.add(n)
            if context_lines:
                wanted.update(
                    range(max(1, n - context_lines), min(len(lines), n + context_lines) + 1)
                )
        for n in sorted(wanted):
            kind = "m" if n in hit_lines else "c"
            records.append(_match_record(rel, n, lines[n - 1], kind))
        if sum(1 for r in records if r[3] == "m") >= GREP_SPILL_MATCH_LIMIT:
            break
    return records, files_searched, files_skipped


_RG_LINE_RE = re.compile(r"^(?P<path>.+?):(?P<line>\d+):(?P<text>.*)$")
_RG_CONTEXT_RE = re.compile(r"^(?P<path>.+?)-(?P<line>\d+)-(?P<text>.*)$")


async def _ripgrep_scan(
    pattern: str,
    target: Path,
    base: Path,
    include: str | None,
    case: bool,
    context_lines: int,
    signal: AbortSignal | None,
) -> tuple[list[tuple[str, int, str, str]], int] | None:
    """Native ripgrep scan; None means "fall back to the Python engine".

    rg's defaults already match this tool's contract — hidden files skipped,
    .gitignore respected, binary files detected — and ``-g '!dir'`` adds the
    fixed vendor prune list the Python walker enforces. The subprocess is
    raced against the abort signal and the same 30s wall clock; output is
    capped well past the spill limit so a pathological tree cannot stream
    forever."""
    rg = _rg_binary()
    if rg is None:
        return None
    rel_target = (target.relative_to(base).as_posix() if base in target.parents else ".") or "."
    # -H forces the path prefix even for a single explicitly-named file
    # (rg otherwise prints bare "lineno:text", which this parser and the
    # shared path:line:text grammar both expect to carry a path).
    argv = [rg, "--no-heading", "--color", "never", "--max-filesize", "1M", "-n", "-H"]
    if not case:
        argv.append("-i")
    if context_lines:
        argv += ["-C", str(context_lines)]
    if include:
        argv += ["-g", include]
    for pruned in _GREP_PRUNE_DIRS:
        argv += ["-g", f"!{pruned}"]
    argv += ["--", pattern, rel_target]

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(base),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except (OSError, ValueError):
        return None

    records: list[tuple[str, int, str, str]] = []
    deadline = time.monotonic() + GREP_SCAN_DEADLINE_S
    output_cap = (GREP_SPILL_MATCH_LIMIT + 1) * 3 + 1000
    try:
        assert proc.stdout is not None
        while len(records) < output_cap:
            if signal and signal.aborted:
                proc.kill()
                return None
            if time.monotonic() > deadline:
                proc.kill()
                break
            try:
                raw = await asyncio.wait_for(proc.stdout.readline(), timeout=1.0)
            except (TimeoutError, asyncio.TimeoutError):
                continue
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            if line == "--":
                continue
            match = _RG_LINE_RE.match(line)
            if match:
                path = match.group("path")
                if path.startswith("./"):
                    path = path[2:]
                records.append(
                    _match_record(path, int(match.group("line")), match.group("text"), "m")
                )
                continue
            context = _RG_CONTEXT_RE.match(line)
            if context:
                path = context.group("path")
                if path.startswith("./"):
                    path = path[2:]
                records.append(
                    _match_record(path, int(context.group("line")), context.group("text"), "c")
                )
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except (TimeoutError, asyncio.TimeoutError):
            proc.kill()
    finally:
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
    # exit 0 = matches, 1 = none; anything else is an rg-side error (bad
    # flag, unreadable cwd) the caller should not inherit.
    if proc.returncode not in (0, 1):
        return None
    match_count = sum(1 for r in records if r[3] == "m")
    return records, match_count


def _render_grep_body(
    records: list[tuple[str, int, str, str]],
    skip: int,
    context_lines: int,
) -> tuple[str, int, int]:
    """``(body, shown_matches, total_matches)`` with skip + display cap applied.

    Matches paginate (``skip`` then the 200-line display cap); context rides
    along only for the matches being shown, groups separated by ``--``."""
    match_indexes = [i for i, r in enumerate(records) if r[3] == "m"]
    total = len(match_indexes)
    kept = match_indexes[skip:][:GREP_MATCH_LIMIT]
    if not kept:
        return "", 0, total
    keep_set = set(kept)
    # Pull in the context records neighbouring each kept match (same file,
    # within ±context_lines of the match).
    if context_lines:
        for i in kept:
            rel, lineno = records[i][0], records[i][1]
            j = i - 1
            while (
                j >= 0
                and records[j][0] == rel
                and records[j][3] == "c"
                and lineno - records[j][1] <= context_lines
            ):
                keep_set.add(j)
                j -= 1
            j = i + 1
            while (
                j < len(records)
                and records[j][0] == rel
                and records[j][3] == "c"
                and records[j][1] - lineno <= context_lines
            ):
                keep_set.add(j)
                j += 1
    out: list[str] = []
    last_line = None
    last_rel = None
    for i in sorted(keep_set):
        rel, lineno, text, kind = records[i]
        if last_rel is not None and (
            last_line is None or rel != last_rel or lineno != last_line + 1
        ):
            out.append("--")
        out.append(f"{rel}:{lineno}:{text}" if kind == "m" else f"{rel}:{lineno}-{text}")
        last_rel, last_line = rel, lineno
    return "\n".join(out), len(kept), total


@_guard("grep")
async def execute_grep(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Regex search over files — ripgrep when available, Python otherwise."""
    try:
        params = GrepParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "grep", exc)
    try:
        regex = re.compile(params.pattern, 0 if params.case else re.IGNORECASE)
    except re.error as exc:
        return _error(tool_call_id, "grep", f"invalid regex '{params.pattern}': {exc}")

    cwd = _safe_cwd(context)
    target, inside, resolvable = _resolve_workspace_path(params.path, cwd)
    if not target.exists():
        return _error(tool_call_id, "grep", f"Path does not exist: {target}")

    # Outside-workspace searches escalate to an approval prompt regardless
    # of the read tier auto-approval the host normally applies.
    if not inside:
        description = _approval_description(target, inside, "grep", resolvable)
        if not await _check_approval(context, "read", description):
            return _error(tool_call_id, "grep", "User declined to search this path.")

    records: list[tuple[str, int, str, str]] = []
    files_searched = 0
    files_skipped = 0
    engine_note = ""

    # Engine choice needs only a stat of an explicitly named file, never the
    # tree walk: the walk is deferred into the worker threads below so the
    # event loop — which is also the TUI's render loop, and under Textual's
    # eager task factory executes every runner in a batch synchronously up to
    # its first true suspension — never pays a directory tree's worth of
    # scandir/stat calls. Walking here inline was the observed freeze on
    # concurrent grep/glob batches (main thread pinned in os_lstat/os_scandir
    # under task_eager_start, one runner at a time, no frame in between).
    target_is_file = target.is_file()
    base = target.parent if target_is_file else target
    use_rg = _use_ripgrep()
    if use_rg and target_is_file:
        try:
            if target.stat().st_size > GREP_FILE_LIMIT_BYTES:
                use_rg = False
        except OSError:
            use_rg = False

    scan_result = None
    if use_rg:
        scan_result = await _ripgrep_scan(
            params.pattern,
            target,
            base,
            params.include,
            params.case,
            params.context_lines,
            signal,
        )
    if scan_result is not None:
        records, _count = scan_result
        engine_note = " (ripgrep)"
        # rg applies --max-filesize silently; recover the count the footer
        # contract promises. The walk+stat pass is the same filesystem load
        # as the scan itself, so it rides a thread raced against abort.
        # Skipped for a single named file: the engine gate above only keeps
        # rg when that file is already under the cap, so the count is
        # definitionally zero and a worker-thread hop to confirm it would be
        # pure overhead on the most common grep shape (review F2).
        if not target_is_file:
            skipped_count, aborted = await _run_with_abort(
                asyncio.to_thread(_count_oversized_files, target),
                signal,
                lambda: None,
            )
            if aborted:
                return _error(tool_call_id, "grep", "Search aborted.")
            # Non-None whenever aborted is False: _run_with_abort returns the
            # coroutine's result on the non-abort arms, and the counter always
            # returns an int. The assert states that contract for the type
            # checker instead of an `or 0` that would silently launder a
            # future internal failure into a zero (review N1).
            assert skipped_count is not None
            files_skipped = skipped_count
    else:
        if signal and signal.aborted:
            return _error(tool_call_id, "grep", "Search aborted.")

        # The walk and the scan are FILESYSTEM + REGEX work on
        # model-controlled input; running either on the event loop would pin
        # the CPU on a backtracking pattern or a large tree and make Ctrl+C
        # unprocessable. Both run in one worker-thread hop raced against the
        # abort signal, with a wall-clock cap bounding the
        # pathological-regex case (regexes are not classified).
        def _walk_and_scan() -> tuple[list[tuple[str, int, str, str]], int, int]:
            files, scan_base = _grep_file_set(target)
            return _python_grep_scan(files, scan_base, regex, params.include, params.context_lines)

        py_result, aborted = await _run_with_abort(
            asyncio.to_thread(_walk_and_scan),
            signal,
            lambda: None,
        )
        if aborted:
            return _error(tool_call_id, "grep", "Search aborted.")
        # Non-None whenever aborted is False (same contract as above): an
        # exception inside the walk/scan propagates to _guard rather than
        # returning None, so a None here could only mean a broken
        # _run_with_abort — assert, never mislabel it "Search aborted."
        # (review F1).
        assert py_result is not None
        records, files_searched, files_skipped = py_result

    matches = [r for r in records if r[3] == "m"]
    if not matches:
        skipped_note = (
            f" ({files_skipped} file(s) skipped over the 1MB cap)" if files_skipped else ""
        )
        where = f"in {files_searched} file(s)" if not engine_note else ""
        return _text(
            tool_call_id,
            "grep",
            f"No matches for '{params.pattern}'{where}{skipped_note}{engine_note}.",
            useless=True,
            details={"useless": True},
        )

    body, shown, total = _render_grep_body(records, params.skip, params.context_lines)
    # The spill holds every MATCH line (context excluded — it is render-time
    # decoration, recoverable by re-running with a narrower pattern/range).
    spill_lines = [f"{rel}:{lineno}:{text}" for rel, lineno, text, _kind in matches]
    shown_block = body
    # The join and spill ride a thread: multi-MB string joins plus the
    # spill's disk write were the loop-bound tail that froze the frame when
    # several searches settled together.
    body_text, spill_details = await asyncio.to_thread(
        _spilled_list_body, spill_lines, shown_block, "grep", context
    )
    header = f"{shown} match(es) for '{params.pattern}'"
    if total > shown:
        header += f" of {total}{'+' if total >= GREP_SPILL_MATCH_LIMIT else ''}"
        header += f" (use skip={params.skip + shown} for the next page)"
    if params.skip:
        header += f" (skipped {params.skip})"
    if files_skipped:
        header += f" ({files_skipped} file(s) skipped over the 1MB cap)"
    return _text(
        tool_call_id, "grep", header + ":\n" + body_text + engine_note, details=spill_details
    )


def build_grep_tool() -> AgentTool:
    return AgentTool(
        name="grep",
        label="Grep",
        description=(
            "Regex search across files ('path:line:text' matches; context_lines "
            "adds surrounding lines, skip paginates; gitignore respected, "
            "ripgrep-fast when installed)."
        ),
        parameters=GrepParams.model_json_schema(),
        approval_tier="read",
        # Read-only search; parallel greps are the common batch shape.
        concurrency="shared",
        interruptible=False,
        execute=execute_grep,
    )


# ---------------------------------------------------------------------------
# todo
# ---------------------------------------------------------------------------

#: The persisted shape. An owner maps to a list of PHASES, each phase a
#: ``{"name", "items"}`` dict whose ``items`` are the SAME item dicts the tool
#: has always used (``{"text", "status"[, "reason"]}``). Keeping the item dict
#: unchanged is the back-compat lever: ``_match_todos``, ``_todo_rows``,
#: ``_TODO_MARKS`` and the panel's row builder all still operate on one item
#: dict once a phase's ``items`` list is in hand. A flat ``init`` becomes ONE
#: implicit phase named ``"Todos"`` (rendered headerless, so an existing
#: caller sees the identical panel it saw before phases existed).
TodoItem = dict[str, str]  # {"text", "status"[, "reason"]} — UNCHANGED
TodoPhase = dict[str, Any]  # {"name": str, "items": list[TodoItem]}

#: The sentinel name of the implicit phase a flat ``init`` (or a bare ``add``)
#: writes into. The panel special-cases exactly this single-phase case to stay
#: headerless, so the constant is the one place that spelling lives.
_IMPLICIT_PHASE = "Todos"

#: In-memory todo lists keyed by NON-EMPTY session id. The host may attach a
#: durable store to the ToolContext (``todos`` dict) — we prefer that so
#: transcripts can replay todo state — but a bare context still works via
#: this table (keyed by the context object's id when no session id exists).
TODO_STORE: dict[str, list[TodoPhase]] = {}
#: Fallback store for contexts without a session id, so their lists never
#: collide under the shared "" key. Keyed by the context object's id rendered
#: as a string, so every todo store in this module has one key type.
_CONTEXT_TODO_STORE: dict[str, list[TodoPhase]] = {}

#: The custom-message type the session's continuation guardrail injects at the
#: yield boundary (``Session._todo_continuation``). It lives beside the store
#: because the todo feature owns the vocabulary and session.py imports it —
#: the same shape as ``HUB_MESSAGE_TYPE`` (harness/comms.py) and
#: ``WAKE_PROMPT_MESSAGE_TYPE`` (harness/wake.py), neither of which is defined
#: in the session that renders them.
TODO_REMINDER_MESSAGE_TYPE = "todo_reminder"

#: Statuses that no longer need work. ``blocked`` is NOT here: a blocked item
#: is unfinished work waiting on someone, and counting it as progress would
#: let a stalled list read as a finished one.
_TODO_RESOLVED = ("done", "dropped")

#: Checkbox marks per status, shared by ``view`` and the error path. The TUI
#: panel renders the same four marks (tui/widgets/todo_panel.py) so the
#: transcript receipt and the dock band cannot describe one list differently.
_TODO_MARKS = {"pending": " ", "done": "x", "blocked": "~", "dropped": "-"}


class InitPhase(BaseModel):
    # A single named group in a phased ``init`` payload. Extra keys fail loud
    # like every other tool param model, so a mis-shaped phase is a validation
    # error the model sees immediately rather than a silently-dropped field.
    model_config = ConfigDict(extra="forbid")

    phase: str = Field(
        description=(
            "phase name — a short noun phrase, e.g. 'Foundation', 'Auth', "
            "'Verification'. No '1.'/'Phase 1:' prefixes; the panel numbers "
            "phases itself."
        )
    )
    items: list[str] = Field(description="task texts for this phase, in order")


class TodoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["init", "add", "done", "block", "drop", "view"] = Field(
        description=(
            "init: replace the whole list, optionally grouped into named phases "
            "(pass `phases`); add: append newly discovered work without "
            "rewriting the list, optionally into a named `phase`; done: mark "
            "items finished; block: "
            "mark items that cannot proceed until a user decides or an "
            "external service answers (requires 'reason'); drop: abandon "
            "items that are no longer needed; view: show the list."
        )
    )
    items: list[str] = Field(
        default_factory=list,
        description=(
            "Todo texts. Required for every op except 'view'. For "
            "done/block/drop each entry must repeat an existing item's text "
            "exactly; several items may be resolved in one call."
        ),
    )
    # NEW — phased init payload. Mutually exclusive with a flat ``items`` on
    # 'init': the two express the same list two ways, so accepting both would
    # be ambiguous about which wins.
    phases: list[InitPhase] = Field(
        default_factory=list,
        description=(
            "Phased task list for 'init': each entry groups task texts under a "
            "named phase. Use this OR 'items' (flat), not both. Phases render "
            "as headers with their items indented beneath."
        ),
    )
    # NEW — phase target for add/done/drop/block. For 'add' it names the phase
    # to append into (lazily created if new); for done/drop/block WITHOUT
    # 'items' it resolves every open item in that phase at once.
    phase: str = Field(
        default="",
        description=(
            "Phase to address. For 'add', append into this phase (lazily "
            "created if new). For 'done'/'drop'/'block' with no 'items', "
            "resolve EVERY open item in this phase. Omit to target the whole "
            "list (add \u2192 implicit 'Todos' phase)."
        ),
    )
    reason: str = Field(
        default="",
        description=(
            "Required for 'block', ignored otherwise: what the item is waiting "
            "on (the decision the user must make, the service that is down). "
            "Use 'drop' instead when the work is simply not needed any more."
        ),
    )


#: Every todo store — host-attached or module-level — maps one owner key to
#: that owner's list of PHASES (``{"name", "items"}``); each phase's ``items``
#: are ``{"text", "status"[, "reason"]}`` item dicts. A flat ``init`` still
#: works: it becomes one implicit ``"Todos"`` phase (see :func:`_as_phases`).
TodoStore = dict[str, list[TodoPhase]]


def _as_phases(raw: list[Any]) -> list[TodoPhase]:
    """Coerce a stored owner-list to phases — the ONE shape every reader walks.

    A legacy flat list (item dicts at the top level) becomes one implicit
    ``"Todos"`` phase; an already-phased list passes through untouched. This is
    defensive rather than required: the store is process-global and in-memory
    (nothing serialises it), and within one process only the post-upgrade tool
    writes it — so the only way a reader meets a legacy flat list is a
    hand-attached ``ToolContext.todos`` holding old data, which must not crash
    a reader. Detects phased by the presence of ``"items"`` on the first
    element, since an item dict never carries that key.
    """
    if raw and isinstance(raw[0], dict) and "items" in raw[0]:
        return raw  # already phased
    return [{"name": _IMPLICIT_PHASE, "items": list(raw)}]  # legacy flat


def _all_items(phases: list[TodoPhase]) -> list[TodoItem]:
    """Every item across phases in phase-then-item order — the flat view the
    match/progress/fingerprint helpers walk.

    Items are shared BY REFERENCE with their phase, so mutating one here (e.g.
    ``item["status"] = "done"``) updates its phase in place, which is what lets
    ``done``/``drop``/``block`` search across phases without re-indexing.
    """
    return [item for phase in phases for item in phase["items"]]


def _todo_store_and_key(context: ToolContext | None) -> tuple[TodoStore, str]:
    """Resolve ``(store, key)`` for this context. An attached ``todos`` dict
    wins; otherwise the module table keyed by session id; a context with NO
    session id gets its own slot keyed by object id, never the shared "" key.
    """
    if context is not None and context.todos is not None:
        # A host-attached store wins even without a session id; the context's
        # own identity keys the slot so bare contexts never share one.
        return context.todos, context.session_id or str(id(context))
    if context is not None and context.session_id:
        return TODO_STORE, context.session_id
    return _CONTEXT_TODO_STORE, str(id(context))


def open_todos(session_id: str) -> list[dict[str, str]]:
    """Copies of the ``pending`` items for ``session_id`` (``[]`` when there
    are none, or the id is unknown).

    THE single definition of "open" that the session's continuation guardrail
    (``Session._todo_continuation``) fires on, so the tool and the guardrail
    cannot disagree about whether a turn may end. ``blocked`` is excluded on
    purpose: it is the escape hatch that lets a model stop honestly instead of
    marking work done it did not do, and nudging it would defeat that.

    Mirrors only the SESSION-ID branch of :func:`_todo_store_and_key`. A host
    that attaches its own store to ``ToolContext.todos`` (harness/types.py:449;
    ``Session._build_tool_context`` never does) writes there instead and would
    have to feed the guardrail itself. Copies because the ops mutate their item
    dicts in place, and a caller holding the originals would read a list that
    changed under it.

    Pending items are flattened across ALL phases (phase-then-item order): the
    guardrail asks "is there open work?", and a phase boundary does not change
    the answer. The returned dicts are plain item dicts (no phase key), so
    ``_todo_reminder_text`` needs no change.
    """
    raw = TODO_STORE.get(session_id)
    if not raw:
        return []
    return [
        dict(item)
        for phase in _as_phases(raw)
        for item in phase["items"]
        if item.get("status") == "pending"
    ]


def todo_snapshot(session_id: str) -> list[dict[str, Any]]:
    """A deep copy of the FULL PHASED todo list for ``session_id`` (``[]`` none).

    The durable form the session persists to its transcript so a resume can
    rebuild the list (see ``Session._persist_todo_snapshot``). Unlike
    :func:`open_todos` this keeps EVERY item — done, dropped, blocked, pending —
    and every field (including a blocked item's ``reason``), because the panel
    and the progress counter render the whole list, not just the open subset.

    Returns the PHASED shape (``[{"name", "items":[{...}]}]``): the store is
    phased (phased-todos change), so persistence must round-trip phases or a
    resume would flatten a multi-phase plan into one anonymous list. The copy is
    DEEP through the phase's ``items`` — a shallow ``dict(phase)`` would alias
    the nested item dicts the tool mutates in place, and a caller holding the
    snapshot would then see ``done`` flip under it. Runs ``_as_phases`` so a
    legacy flat slot still snapshots as one implicit phase.

    Reads the same SESSION-ID branch of the store as :func:`open_todos`; the
    caveat there about a host-attached ``ToolContext.todos`` store applies here
    identically.
    """
    raw = TODO_STORE.get(session_id)
    if not raw:
        return []
    return [
        {"name": phase["name"], "items": [dict(item) for item in phase["items"]]}
        for phase in _as_phases(raw)
    ]


def restore_todos(session_id: str, phases: list[dict[str, Any]]) -> None:
    """Seed ``session_id``'s todo list from a persisted snapshot at resume.

    The inverse of :func:`todo_snapshot`. Writes straight into the same
    module-level table the todo tool uses, so the restored list is
    indistinguishable from one the tool built — the panel reads it, and the
    continuation guardrail's fingerprint (:func:`todo_fingerprint`) matches what
    it was before the restart.

    Accepts the PHASED snapshot and normalises it through :func:`_as_phases`, so
    a snapshot written by the pre-phases build (a flat item list) still restores
    as one implicit phase rather than corrupting the store. The deep copy
    mirrors :func:`todo_snapshot`: each phase gets a fresh ``items`` list of
    copied item dicts so the store never aliases the caller's structure.

    An empty snapshot is authoritative too: keeping its empty slot distinguishes
    a restored clear from a child whose plan has not been loaded yet. Existing
    slots, including empty ones, win over a stale on-demand read that raced a
    live tool call. An empty session id is never a store identity.
    """
    if not session_id or session_id in TODO_STORE:
        return
    if not phases:
        TODO_STORE[session_id] = []
        return
    TODO_STORE[session_id] = [
        {"name": phase["name"], "items": [dict(item) for item in phase["items"]]}
        for phase in _as_phases(phases)
    ]


def todo_fingerprint(session_id: str) -> tuple[tuple[str, str, str], ...]:
    """``(phase_name, text, status)`` for EVERY item, phase-then-item order.

    What the guardrail's no-progress latch compares between two yields. The
    FULL list, not the pending subsequence: an item moving from ``done`` to
    ``dropped`` changes only the settled part of the list, and a latch blind to
    that would call a list that did move unchanged. The phase name rides in the
    tuple because with phases a rename or an item moving between phases is also
    movement the model made — a 2-tuple would be blind to it and read a list
    that changed as stuck.

    CROSS-FILE COUPLING (design §5.3): the arity of this tuple is mirrored by
    ``session.py:_stamped_todo_fingerprint``, which filters the stamped copy to
    the SAME length. If one grows without the other, the stamped side becomes
    empty and every reminder expires on every render — the latch errs safe
    (keeps nudging) so 'does it nudge' still passes while 'does it stop
    nudging' silently breaks. Change both together.

    Store knowledge stays in this module — :func:`open_todos`'s note about a
    host-attached ``ToolContext.todos`` store applies here identically.
    """
    raw = TODO_STORE.get(session_id)
    if not raw:
        return ()
    return tuple(
        (str(phase["name"]), str(item.get("text", "")), str(item.get("status", "pending")))
        for phase in _as_phases(raw)
        for item in phase["items"]
    )


def _todo_progress(current: list[dict[str, str]]) -> str:
    """``n/total`` counting RESOLVED items (done or dropped).

    The TUI panel header counts the same way, so the receipt the model reads
    and the band the user reads describe one list identically.
    """
    resolved = sum(1 for item in current if item.get("status") in _TODO_RESOLVED)
    return f"{resolved}/{len(current)}"


def _todo_rows(items: list[dict[str, str]]) -> list[str]:
    """One ``- [mark] text`` row per item, blocked rows carrying their reason."""
    rows: list[str] = []
    for item in items:
        status = item.get("status", "pending")
        row = f"- [{_TODO_MARKS.get(status, ' ')}] {item['text']}"
        if status == "blocked":
            reason = item.get("reason", "")
            row += f" — blocked: {reason}" if reason else " — blocked"
        rows.append(row)
    return rows


def _todo_view_text(phases: list[TodoPhase]) -> str:
    """The ``view`` receipt, grouped by phase so it mirrors the dock panel.

    A single implicit ``"Todos"`` phase renders HEADERLESS — byte-identical to
    the pre-phases flat output — so an existing caller (and every existing
    test) reads the exact receipt it always did. Any other shape (multiple
    phases, or one explicitly-named phase) prefixes each group with a
    ``PhaseName · done/total`` header, counting resolved items the same way
    :func:`_todo_progress` does so the header and the panel agree.

    The header spelling is the middot ``PhaseName · done/total`` — the SAME
    grammar the dock panel's ``_phase_header_row`` paints (U5). The receipt used
    to parenthesise (``Foundation (1/2)``) while the panel used the middot, so a
    user cross-referencing the transcript against the band saw two formats for
    one fact; the design (§5.2) makes the panel/receipt mirror load-bearing, so
    both surfaces now spell a phase header one way.
    """
    if len(phases) == 1 and phases[0]["name"] == _IMPLICIT_PHASE:
        return "\n".join(_todo_rows(phases[0]["items"]))
    blocks: list[str] = []
    for phase in phases:
        items = phase["items"]
        blocks.append(f"{phase['name']} · {_todo_progress(items)}")
        blocks.extend(_todo_rows(items))
    return "\n".join(blocks)


def _find_or_create_phase(phases: list[TodoPhase], name: str) -> TodoPhase:
    """The phase named ``name``, created empty and appended if it does not yet
    exist — the lazy-create ``add`` relies on so a new phase name is not an
    error but a new group. Match is by exact name, the same identity the
    fingerprint and panel use.
    """
    for phase in phases:
        if phase["name"] == name:
            return phase
    phase = {"name": name, "items": []}
    phases.append(phase)
    return phase


def _match_todos(
    current: list[dict[str, str]], texts: list[str], *, target: str
) -> tuple[list[dict[str, str]], list[str]]:
    """Resolve item texts to items: ``(matched, missing_texts)``.

    One lookup shared by done/block/drop so the three cannot drift in what
    counts as a match. Exact text match (the model is echoing text it was
    handed); among same-text duplicates the first item not ALREADY in the
    target status wins, so re-issuing an op is idempotent instead of an error
    — a retried tool call must not read as a mistake the model then tries to
    correct.
    """
    matched: list[dict[str, str]] = []
    missing: list[str] = []
    for text in texts:
        candidates = [item for item in current if item["text"] == text]
        if not candidates:
            missing.append(text)
            continue
        matched.append(
            next((item for item in candidates if item.get("status") != target), candidates[0])
        )
    return matched, missing


def _todo_miss_error(
    tool_call_id: str,
    op: str,
    applied: list[dict[str, str]],
    missing: list[str],
    current: list[dict[str, str]],
) -> ToolResult:
    """The partial-match failure: name the texts that missed AND show what is
    still open.

    Whatever DID match stays applied — real progress must not be rolled back
    because one text was mistyped. The open items ride along so the model can
    correct itself in its next call instead of spending a round trip on
    ``view``.
    """
    lines = []
    if applied:
        lines.append(f"Applied '{op}' to: {', '.join(item['text'] for item in applied)}.")
    lines.append(f"No todo matching: {', '.join(repr(text) for text in missing)}.")
    still_open = [item for item in current if item.get("status") in ("pending", "blocked")]
    if still_open:
        lines.append("Open items:")
        lines.extend(_todo_rows(still_open))
    else:
        lines.append("No open items remain.")
    return _error(tool_call_id, "todo", "\n".join(lines))


@_guard("todo")
async def execute_todo(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Maintain a visible task list so progress survives compaction."""
    try:
        params = TodoParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "todo", exc)
    store, key = _todo_store_and_key(context)

    def changed() -> None:
        callback = context.on_todos_changed if context is not None else None
        if callback is not None:
            callback()

    if params.op == "init":
        # ``phases`` and flat ``items`` are two spellings of the same list, so
        # accepting both is ambiguous about which wins — reject it loudly.
        if params.phases and params.items:
            return _error(
                tool_call_id,
                "todo",
                "'init' takes `phases` OR `items`, not both.",
            )
        if params.phases:
            # Each declared phase becomes a group; its texts become pending
            # item dicts of the same shape every other op already handles.
            total = sum(len(phase.items) for phase in params.phases)
            if not total:
                return _error(tool_call_id, "todo", "'init' requires at least one item")
            store[key] = [
                {
                    "name": phase.phase,
                    "items": [{"text": text, "status": "pending"} for text in phase.items],
                }
                for phase in params.phases
            ]
            changed()
            return _text(
                tool_call_id,
                "todo",
                f"Todo list initialized with {total} item(s) across "
                f"{len(params.phases)} phase(s).",
            )
        if not params.items:
            return _error(tool_call_id, "todo", "'init' requires a non-empty items list")
        # Flat init → one implicit headerless phase, so an existing caller sees
        # the identical list it always did (design §3.2, the back-compat lever).
        store[key] = [
            {
                "name": _IMPLICIT_PHASE,
                "items": [{"text": item, "status": "pending"} for item in params.items],
            }
        ]
        changed()
        return _text(
            tool_call_id,
            "todo",
            f"Todo list initialized with {len(params.items)} item(s).",
        )

    phases = _as_phases(store.get(key, []))
    # The flat view across all phases; items are shared BY REFERENCE with their
    # phase, so mutating one here updates the phase in place (design §4.2).
    current = _all_items(phases)

    if params.op == "add":
        if not params.items:
            return _error(tool_call_id, "todo", "'add' requires a non-empty items list")
        # Append into the named phase (default the implicit ``"Todos"``),
        # lazily creating it so a fresh phase name is a new group rather than an
        # error. Dedupe-by-open-text is applied WITHIN the target phase only:
        # the same text living pending in another phase is a legitimately
        # different task, and collapsing across phases would silently drop it.
        target_phase = _find_or_create_phase(phases, params.phase or _IMPLICIT_PHASE)
        open_texts = {
            item["text"] for item in target_phase["items"] if item.get("status") == "pending"
        }
        added: list[str] = []
        for text in params.items:
            if text in open_texts:
                continue
            open_texts.add(text)
            added.append(text)
        target_phase["items"].extend({"text": text, "status": "pending"} for text in added)
        # Reassign: ``store.get(key, [])`` hands back a fresh list when this
        # owner has no list yet, so an `add` before any `init` must bind it.
        store[key] = phases
        current = _all_items(phases)
        if not added:
            return _text(
                tool_call_id,
                "todo",
                f"Already tracked, nothing added ({_todo_progress(current)} resolved).",
            )
        changed()
        return _text(
            tool_call_id,
            "todo",
            f"Added {len(added)} item(s): {', '.join(added)} "
            f"({_todo_progress(current)} resolved).",
        )

    if params.op in ("done", "block", "drop"):
        reason = params.reason.strip()
        if params.op == "block" and not reason:
            return _error(
                tool_call_id,
                "todo",
                "'block' requires a non-empty reason: without it a blocked item "
                "is indistinguishable from abandoned work. Say what it waits on, "
                "or use 'drop' if it is no longer needed.",
            )
        target = {"done": "done", "block": "blocked", "drop": "dropped"}[params.op]
        verb = {"done": "Marked done", "block": "Blocked", "drop": "Dropped"}[params.op]

        if not params.items:
            # Phase-target form: resolve EVERY currently-open item in the named
            # phase at once. Idempotent by construction — a fully-resolved phase
            # selects nothing and reports it cleanly rather than erroring.
            if not params.phase:
                return _error(
                    tool_call_id,
                    "todo",
                    f"'{params.op}' requires either items with the item text or a "
                    f"'phase' to resolve every open item in.",
                )
            named = next((p for p in phases if p["name"] == params.phase), None)
            if named is None:
                return _error(
                    tool_call_id,
                    "todo",
                    f"No phase named {params.phase!r}.",
                )
            matched = [
                item for item in named["items"] if item.get("status") in ("pending", "blocked")
            ]
            for item in matched:
                item["status"] = target
                if target == "blocked":
                    item["reason"] = reason
                else:
                    item.pop("reason", None)
            if not matched:
                return _text(
                    tool_call_id,
                    "todo",
                    f"No open items in phase {params.phase!r} "
                    f"({_todo_progress(current)} resolved).",
                )
            text = f"{verb}: {', '.join(item['text'] for item in matched)}"
            if target == "blocked":
                text += f" — reason: {reason}"
            changed()
            return _text(tool_call_id, "todo", f"{text} ({_todo_progress(current)} resolved).")

        # Text form: search across ALL phases (the flat ``current`` view), so a
        # model echoing a text need not know which phase holds it.
        matched, missing = _match_todos(current, params.items, target=target)
        for item in matched:
            item["status"] = target
            if target == "blocked":
                item["reason"] = reason
            else:
                # A resolved item keeps no stale blocker: the reason described
                # a wait that is over, and the panel would still render it.
                item.pop("reason", None)
        if matched:
            changed()
        if missing:
            return _todo_miss_error(tool_call_id, params.op, matched, missing, current)
        text = f"{verb}: {', '.join(item['text'] for item in matched)}"
        if target == "blocked":
            text += f" — reason: {reason}"
        return _text(tool_call_id, "todo", f"{text} ({_todo_progress(current)} resolved).")

    # op == "view"
    if not current:
        return _text(
            tool_call_id,
            "todo",
            "No todos recorded yet.",
            useless=True,
            details={"useless": True},
        )
    return _text(tool_call_id, "todo", _todo_view_text(phases))


def build_todo_tool() -> AgentTool:
    return AgentTool(
        name="todo",
        label="Todo",
        description=(
            "Track a visible task list (init / add / done / block / drop / view), "
            "optionally grouped into named phases."
        ),
        parameters=TodoParams.model_json_schema(),
        # read tier exemption: todo mutates only session-local bookkeeping
        # (no files, no autonomous turns), so it stays auto-approved.
        approval_tier="read",
        # init rewrites the whole list and add appends to it; concurrent calls
        # would lose one, so the tool runs exclusive despite being cheap.
        concurrency="exclusive",
        interruptible=False,
        execute=execute_todo,
    )


# ---------------------------------------------------------------------------
# wake
# ---------------------------------------------------------------------------


class WakeParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["create", "list", "cancel"] = Field(
        description="create: schedule a wake; list: show schedules; cancel: remove one."
    )
    message: str | None = Field(
        default=None, description="Message delivered when the wake fires (create only)."
    )
    # Scheduling (create) — one of `in` or `at` selects the first due time.
    field_in: str | None = Field(
        default=None,
        alias="in",
        description="Delay before first fire: '45s'|'30m'|'2h'|'8h30m'|'7d'|'1w' "
        "(compound terms like '1h30m' sum).",
    )
    at: str | None = Field(
        default=None,
        description="First fire time: 'HH:MM', '+<duration>', or ISO datetime.",
    )
    every: str | None = Field(
        default=None,
        description="Repeat interval duration, e.g. '1h' or compound '1h30m'.",
    )
    until: str | None = Field(default=None, description="Retire after this time (ISO datetime).")
    limit: int | None = Field(default=None, ge=1, description="Max number of fires.")
    id: str | None = Field(default=None, description="Schedule id (cancel; from wake list).")


def _wake_due_label(schedule: WakeSchedule) -> str:
    due = datetime.fromtimestamp(schedule.next_due_at / 1000, tz=UTC)
    every = f" every {format_duration(schedule.every_ms)}" if schedule.every_ms else ""
    fired = f" (fired {schedule.fired_count}x)" if schedule.fired_count else ""
    return f"next at {due.isoformat()}{every}{fired}"


async def _wake_list(tool_call_id: str, scheduler: WakeSchedulerProtocol) -> ToolResult:
    schedules = list(scheduler.schedules)
    if not schedules:
        return _text(
            tool_call_id,
            "wake",
            "No wake schedules.",
            useless=True,
            details={"useless": True},
        )
    lines = [f'- {s.id}: "{s.message}" {_wake_due_label(s)}' for s in schedules]
    return _text(tool_call_id, "wake", f"{len(schedules)} wake schedule(s):\n" + "\n".join(lines))


async def _wake_create(
    tool_call_id: str, params: WakeParams, scheduler: WakeSchedulerProtocol, now_ms: int
) -> ToolResult:
    existing = list(scheduler.schedules)
    request: dict[str, Any] = {
        "message": params.message or "",
        "in": params.field_in,
        "at": params.at,
        "every": params.every,
        "until": params.until,
        "limit": params.limit,
    }
    outcome = build_wake_schedule(request, existing, now_ms)
    if "error" in outcome:
        return _error(tool_call_id, "wake", outcome["error"])
    schedule = outcome["schedule"]
    updated = [s for s in existing if s.id != schedule.id] + [schedule]
    await scheduler.update(updated)
    due = datetime.fromtimestamp(schedule.next_due_at / 1000, tz=UTC)
    return _text(
        tool_call_id,
        "wake",
        f"Scheduled wake '{schedule.id}' at {due.isoformat()}: \"{schedule.message}\"",
    )


async def _wake_cancel(
    tool_call_id: str, params: WakeParams, scheduler: WakeSchedulerProtocol
) -> ToolResult:
    if not params.id:
        return _error(tool_call_id, "wake", "'cancel' requires the schedule id (see wake list)")
    existing = list(scheduler.schedules)
    remaining = [s for s in existing if s.id != params.id]
    if len(remaining) == len(existing):
        ids = ", ".join(s.id for s in existing) or "none"
        return _error(
            tool_call_id,
            "wake",
            f"No wake schedule with id '{params.id}' (known: {ids})",
        )
    await scheduler.update(remaining)
    return _text(tool_call_id, "wake", f"Cancelled wake schedule '{params.id}'.")


def build_wake_tool(context: ToolContext) -> AgentTool | None:
    """CreateIf builder: the tool only exists when the context carries a wake
    scheduler. A session without wakes must not advertise a tool whose every
    call errors (the createIf convention)."""
    if context.wake_scheduler is None:
        return None
    return AgentTool(
        name="wake",
        label="Wake",
        describe_approval=_describe_wake_approval,
        description="Schedule a future wake (create/list/cancel), e.g. 'in 30m' or 'in 8h30m'.",
        parameters=WakeParams.model_json_schema(),
        # write tier: wake create persists schedules and arms unattended
        # future agent turns — the only tool that creates autonomous
        # execution, so it prompts like a mutation (the loop gates write/exec).
        approval_tier="write",
        # create/cancel rewrite the whole schedule list; two concurrent
        # calls would lose one, so the tool runs exclusive.
        concurrency="exclusive",
        interruptible=False,
        execute=execute_wake,
    )


@_guard("wake")
async def execute_wake(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Create, list, or cancel scheduled wakes via the session's scheduler."""
    try:
        params = WakeParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "wake", exc)
    scheduler = context.wake_scheduler if context else None
    if scheduler is None:
        return _error(
            tool_call_id,
            "wake",
            "Wake scheduling is not available in this session (no scheduler attached).",
        )
    now_ms = int(time.time() * 1000)
    if params.op == "list":
        return await _wake_list(tool_call_id, scheduler)
    if params.op == "create":
        if not params.message or not params.message.strip():
            return _error(tool_call_id, "wake", "'create' requires a non-empty message")
        if not params.field_in and not params.at:
            return _error(tool_call_id, "wake", "'create' requires 'in' or 'at'")
        return await _wake_create(tool_call_id, params, scheduler, now_ms)
    return await _wake_cancel(tool_call_id, params, scheduler)


# ---------------------------------------------------------------------------
# send — hand a message to another local lop session
# ---------------------------------------------------------------------------
#
# Why this is a tool and not a `bash` of ``lop send``: shelling out buries a
# cross-session delivery inside an opaque shell trace (hard to audit, easy to
# miss in approval), and the CLI's DEFAULT is mailbox-only — an idle target
# parked on a scheduled wake then sits on the note until its next wake fires.
# The tool defaults ``wake`` ON so an idle peer answers right away, names the
# delivery in its own card, and prompts at the write tier like every other
# capability that starts autonomous work. Receive semantics live entirely in
# ``Session.receive_peer_message``; this side only resolves, validates and
# dials, through the shared send-side core in ``mobile/peer_send.py``.


class SendParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str | None = Field(
        default=None,
        description=(
            "Peer to message: case-insensitive substring of the conversation name, "
            "session id, or cwd basename (`lop sessions` lists what is running). "
            "ALTERNATIVE to pid/session, not a companion — pass exactly one way of "
            "addressing the peer; target together with pid or session is refused as "
            "an ambiguous recipient."
        ),
    )
    pid: int | None = Field(
        default=None,
        description=("Exact pid of the peer session. Use INSTEAD of target, never alongside it."),
    )
    session: str | None = Field(
        default=None,
        description=("Exact session id of the peer. Use INSTEAD of target, never alongside it."),
    )
    message: str = Field(
        min_length=1,
        description="The message body; it lands in the peer's transcript as an inbound card.",
    )
    wake: bool = Field(
        default=True,
        description=(
            "Mailbox mode: wake an idle peer so it responds right away. Defaults ON; "
            "False is the quiet drop the peer reads on its next turn. Ignored when "
            "now=True."
        ),
    )
    now: bool = Field(
        default=False,
        description=(
            "Steer the peer mid-turn instead of using the mailbox; opens a turn if "
            "the peer is idle."
        ),
    )


#: How a peer send is addressed and how it will land, as the two words both the
#: approval prompt and the TUI card need. ONE definition on purpose: the two
#: surfaces format differently but must never disagree about WHICH peer or WHICH
#: delivery mode a call names, and two copies of that precedence would drift the
#: first time a mode is added (review round 1, NIT-2).
def peer_send_target_label(args: dict[str, Any]) -> str:
    """The addressed peer: ``pid <n>`` / ``session <id>`` / the raw substring.

    Mirrors the resolver's own precedence (pid, then session id, then
    substring). ``?`` when nothing addresses a peer — the call will fail, but the
    row and the prompt are painted before that, and a blank slot reads as though
    the next field were the target.
    """
    pid = args.get("pid")
    session = str(args.get("session") or "").strip()
    target = " ".join(str(args.get("target") or "").split())
    if isinstance(pid, int) and not isinstance(pid, bool):
        return f"pid {pid}"
    if session:
        return f"session {session}"
    return target or "?"


def peer_send_mode_label(args: dict[str, Any]) -> str:
    """The delivery promise in one word: ``now`` / ``quiet`` / ``wake``.

    ``now`` steers the peer mid-turn, ``wake`` (the default) drives an idle
    peer's turn, ``quiet`` (``wake=False``) waits for the peer's next turn.
    """
    if args.get("now"):
        return "now"
    return "quiet" if args.get("wake") is False else "wake"


def _describe_send_approval(args: dict[str, Any], cwd: str) -> str:
    """``to <target> (<mode>): <body>`` — who gets it, how it lands, what it says.

    All three are the decision: waking an idle session and quietly dropping a
    note are different commitments, and ``pid 48213`` versus a substring is the
    difference between one peer and whichever matches.

    The clause does NOT repeat the tool name. The host already prefixes
    ``Allow send?``, which is why every sibling describer supplies its own verb
    (``run:``, ``subagent:``, ``schedule:``, ``browse:``); ``send to …`` under
    that prefix read "Allow send? send to …" (design round 1, D5).

    The body is bounded in CELLS with the app's ellipsis, not in characters with
    ASCII dots: a CJK body clipped by character count measured 138 cells against
    an intended 60 and wrapped the prompt onto a second line (design round 1, D4).
    """
    who = peer_send_target_label(args)
    mode = peer_send_mode_label(args)
    body = _truncate_approval_body(" ".join(str(args.get("message") or "").split()))
    return f"to {who} ({mode}): {body}" if body else f"to {who} ({mode})"


def build_send_tool(context: ToolContext) -> AgentTool | None:
    """Always-on builder: an unconditional factory in the createIf table.

    Unlike ``wake`` or ``browser``, the capability does not depend on an
    attachment THIS session carries — every session sits on the same registry +
    loopback control substrate, and "no peer matches right now" is a per-call
    answer, not a missing capability. Gating it away would strip it from exactly
    the sessions that message peers.
    """
    return AgentTool(
        name="send",
        label="Peer send",
        describe_approval=_describe_send_approval,
        description=(
            "Hand a message to another local lop session on this machine (no cmux). "
            "Address the peer by EXACTLY ONE of `target` (name/cwd substring), `pid` "
            "(exact), or `session` (exact session id) — they are alternatives, and "
            "passing a `target` together with a `pid`/`session` is refused as an "
            "ambiguous recipient rather than resolved. `lop sessions` lists what is "
            "running. By default the message lands in the peer's mailbox AND wakes the peer if it "
            "is idle, so an idle peer responds right away; `wake=False` is the quiet "
            "mailbox drop (read on the peer's next turn), and `now=True` steers "
            "mid-turn (opens a turn if the peer is idle). The result says how the "
            "peer received it."
        ),
        parameters=SendParams.model_json_schema(),
        # write tier: a delivery can start an autonomous turn in ANOTHER session
        # (wake drives an idle peer's turn; now steers or opens one) — the same
        # commitment wake's write tier names, so it prompts like a mutation.
        approval_tier="write",
        # EXCLUSIVE because of the wire, not because of tool state. The shared
        # resource is the PEER's control socket: `send_peer_message` dials
        # daemon-class, and a registrant admits at most one daemon connection —
        # a new daemon dial EVICTS the existing one (`registrant.py`). Two
        # concurrent sends therefore tear down the first sender's socket while
        # it still awaits its ack, so it raises ConnectionError and reports
        # "could not deliver" for a message the peer already received and
        # processed (measured: 3 concurrent sends -> 2 ConnectionErrors, 3/3
        # delivered). A false negative on a delivery receipt is worse than an
        # error, because the model's natural response is to retry and duplicate
        # the message. Serialising within the batch matches the wire's real
        # one-daemon-at-a-time contract; `hub`, the closest sibling, is
        # exclusive for its own reasons. (A non-evicting client class for peer
        # sends would lift this, and is out of scope here.)
        concurrency="exclusive",
        # interruptible: the only wait is the peer's ack under a bounded
        # deadline; cancelling on Esc/steer is free (the frame either acked or
        # it did not) and keeps the turn responsive like the other network tools.
        interruptible=True,
        execute=execute_send,
    )


def _peer_sender_conversation_name(context: ToolContext) -> str:
    """The name a peer's inbound card should show for THIS session.

    ``session_name`` alone is wrong on a subagent. It resolves to the PARENT's
    title (a child has none of its own and can never grow one), so a peer
    message sent by a child presented under its parent's conversation name
    while the ``session_id`` beside it correctly named the child — two
    different sessions in one card. Composing the child's own label onto it is
    the same answer the browser tab pill gives (``<parent> › <label>``), and
    for the same reason: the label is the only name the child actually owns.

    Composed here rather than through :func:`_browser_subagent_label` even
    though the form is identical: that function budgets the result to the
    browser pill's 30 clusters, a constraint a peer card does not have (a
    registry-sourced ``conversation_name`` runs to ``MAX_TITLE_CHARS``), and
    borrowing it would silently ellipsise a name for a surface that had room.

    Display only, on the FALLBACK path alone — a host that published a registry
    record names the sender from that record and never reaches here. Identity
    stays ``session_id``, which is the child's throughout.
    """
    # Both fields, matching the browser side's discriminator: ``job_id`` alone
    # is also carried by server sessions, and only a subagent has a label.
    if context.job_id is None or not context.job_label.strip():
        return context.session_name
    parent = context.session_name.strip()
    label = context.job_label.strip()
    return f"{parent}{_BROWSER_SUBAGENT_SEPARATOR}{label}" if parent else label


@_guard("send")
async def execute_send(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Resolve a peer session, validate the body, and deliver over the loopback
    control socket, returning the receive side's own detail string."""
    try:
        params = SendParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "send", exc)

    from local_operator.mobile.peer_send import (
        candidate_lines,
        peer_sender_identity_async,
        resolve_peer_target,
        validate_peer_body,
    )

    # Off the loop: the resolver reads and parses every registry record, and
    # this tool runs inside the session's own event loop, where a blocking
    # filesystem walk stalls the UI along with every other task.
    record, candidates, error = await asyncio.to_thread(
        resolve_peer_target,
        target=params.target,
        pid=params.pid,
        session=params.session,
    )
    if candidates:
        # ``pid=<n>`` rather than ``pid <n>``: the reader is a model that has to
        # turn this line into an argument, so the line is written in the
        # parameter syntax it will copy (review round 1, MINOR-1).
        #
        # "DROP `target` and retry with" rather than "retry with": the minimal
        # edit a model makes to its previous call is to keep `target` and add
        # `pid`, and that pair is now refused as an ambiguous recipient. The
        # instruction must name the removal explicitly or it teaches the error.
        lines = [
            f"{len(candidates)} sessions match; drop `target` and retry with pid=<n> "
            f"instead (passing both is refused):"
        ]
        lines.extend(candidate_lines(candidates, indent="  ", prefix="pid="))
        return _error(tool_call_id, "send", "\n".join(lines))
    cold_session_id = ""
    if record is None:
        # An exact ``session`` may still name a stored session that is simply
        # not running. A quiet note to one of those is spooled rather than
        # refused (that is what ``wake=false`` asks for); anything wanting
        # attention engages a runtime. See ``peer_send.deliver_peer_message``.
        from local_operator.mobile.peer_send import resolve_cold_session

        cold_session_id = await asyncio.to_thread(resolve_cold_session, params.session or "")
        cold_session_id = cold_session_id or ""
    if not cold_session_id and (error or record is None):
        return _error(tool_call_id, "send", error or "no target resolved")

    # Self-send guard: the tool runs INSIDE the sender's session process, so
    # os.getpid() IS the sending session — the CLI child checks os.getppid()
    # for the same reason (its parent is the session). A target resolving to
    # this pid would paint a "peer message from <own name>" card as though a
    # DIFFERENT session sent it and, with wake/now, self-trigger a turn.
    # Refuse before any dial.
    if record is not None and record.pid == os.getpid():
        return _error(
            tool_call_id,
            "send",
            "that target is this session; a session cannot peer-message itself — "
            "fold the note into your own work instead",
        )

    body_error = validate_peer_body(params.message)
    if body_error:
        return _error(tool_call_id, "send", body_error)

    mode = "steer" if params.now else "mailbox"
    # Also off the loop: the ancestry walk runs a registry scan and a ``ps`` per
    # hop. It matches on the first hop here (the tool IS the session process),
    # but the cost is not structurally bounded and must not sit on the loop.
    sender = await peer_sender_identity_async(os.getpid())
    if "session_id" not in sender and context is not None:
        # No registry record named this process (a reduced host that never
        # published one): fall back to the ToolContext identity so the peer's
        # inbound indicator can still name the sender. The name maps to
        # ``conversation_name`` because that is the key the indicator reads.
        if context.session_id:
            sender["session_id"] = context.session_id
        name = _peer_sender_conversation_name(context)
        if name:
            sender["conversation_name"] = name

    from local_operator.mobile.peer_send import deliver_peer_message

    try:
        # Awaited directly — this execute is already async; an asyncio.run here
        # would try to nest a loop inside the running one.
        detail = await deliver_peer_message(
            record,
            session_id=(record.session_id if record is not None else cold_session_id),
            text=params.message,
            mode=mode,
            wake=bool(params.wake),
            sender=sender,
        )
    except RuntimeError as exc:
        # A protocol-level refusal (an older registrant that does not know the
        # op, a handle that cannot receive): the peer answered, and its answer
        # was no. Nothing was delivered, so the model may safely retry elsewhere.
        return _error(tool_call_id, "send", f"could not deliver: {exc}")
    except (ConnectionError, OSError, ValueError) as exc:
        # The connection or the ack failed — which is NOT the same as the
        # message not arriving. The receive side commits the message before it
        # acks, so a dropped socket or an ack timeout (asyncio.TimeoutError is
        # an OSError subclass) can mean "delivered, receipt lost". Saying
        # "could not deliver" here would assert a non-delivery this side cannot
        # know, and a model that believes it retries and duplicates the message
        # (review round 1, MINOR-3).
        target = (
            f"{record.conversation_name or record.session_id} (pid {record.pid})"
            if record is not None
            else f"{cold_session_id} (not running)"
        )
        return _error(
            tool_call_id,
            "send",
            f"no delivery confirmation from {target}: {exc}. "
            "The message may or may not have arrived — "
            "check with the peer before resending, or it may be delivered twice.",
        )
    if record is not None:
        name = record.conversation_name or record.session_id
        return _text(
            tool_call_id,
            "send",
            f"→ {name} (pid {record.pid}): {detail}",
            details={"pid": record.pid, "mode": mode, "wake": bool(params.wake)},
        )
    # A session with no runtime: the receipt names the session rather than a
    # pid, because there is no process to name and claiming one would be a lie
    # the model might then try to signal.
    return _text(
        tool_call_id,
        "send",
        f"→ {cold_session_id} (not running): {detail}",
        details={
            "session_id": cold_session_id,
            "mode": mode,
            "wake": bool(params.wake),
        },
    )


# ---------------------------------------------------------------------------
# variables — list / read session variables (values never enter the prompt)
# ---------------------------------------------------------------------------


class ListVariablesParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReadVariableParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Variable name to read.")


#: Safety cap on a single variable value returned to the model. Keeps an
#: accidental read of a huge value from blowing up context; oversize values
#: are elided with a marker rather than dumped in full.
MAX_VARIABLE_VALUE_CHARS = 4000


def _variable_store(context: ToolContext | None) -> VariableStoreProtocol:
    """The session's VariableStore, or a fresh env-only store as fallback.

    A session attaches its store (config variables + project file + env) to
    ``context.variables``; when absent (bare tool tests) we fall back to a
    store over the process environment so the tools still answer."""
    if context is not None and context.variables is not None:
        return context.variables
    from local_operator.variables import VariableStore

    return VariableStore(cwd=_safe_cwd(context))


@_guard("list_variables")
async def execute_list_variables(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Return variable NAMES only (never values) so the agent can pick what
    to read without pulling everything into context. One compact line each."""
    try:
        ListVariablesParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "list_variables", exc)
    store = _variable_store(context)
    names = store.names()
    shown = names if len(names) <= 100 else names[:100] + ["…"]
    body = "\n".join(shown) if shown else "(no variables defined)"
    # Session credentials are a SEPARATE namespace from ``names()`` (which
    # excludes them by design — a credential must never resolve through the
    # ordinary read path). But a model that only sees "(no variables)" has
    # no way to know a credential the operator just stored exists, and the
    # live failure (session 835fbcafdc27) was exactly that: list_variables
    # said nothing, the model guessed a conventional name, and ten minutes
    # passed before it found the real one in the system-prompt tail. Names
    # only — the value stays out of tool results entirely.
    credential_names = store.credential_names() if hasattr(store, "credential_names") else []
    if credential_names:
        listed = ", ".join(credential_names)
        body = (
            f"{body}\n\n"
            "session credentials (secret, usable via bash env, not readable): "
            f"{listed}"
        )
    return _text(
        tool_call_id,
        "list_variables",
        f"{len(names)} variable(s) available:\n{body}",
        details={"count": len(names), "credentials": len(credential_names)},
    )


@_guard("read_variable")
async def execute_read_variable(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Read ONE variable value on demand; unknown names return a not-found
    error (the loop surfaces it, the caller can list_variables)."""
    try:
        params = ReadVariableParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "read_variable", exc)
    if not params.name.strip():
        return _error(tool_call_id, "read_variable", "name must be a non-empty string")
    store = _variable_store(context)
    # Credential check BEFORE the unknown-variable branch: a stored credential
    # is deliberately absent from ``names()`` (values must never resolve here),
    # so without this the tool answers "unknown variable (see list_variables)"
    # — circular advice, now that list_variables names credentials — and the
    # model burns a turn retrying a name that can never be read. The error
    # says where the value DOES live instead.
    credential_names = store.credential_names() if hasattr(store, "credential_names") else []
    if params.name in credential_names:
        return _error(
            tool_call_id,
            "read_variable",
            f"SESSION CREDENTIAL: {params.name} is a session credential. Its "
            "value is never readable. It is injected as an environment "
            f"variable into every bash command — use it there (e.g. a child "
            f"process reads ${params.name}), never echo it.",
        )
    if params.name not in store.names():
        return _error(
            tool_call_id, "read_variable", f"unknown variable: {params.name} (see list_variables)"
        )
    try:
        value = store.read(params.name)
    except KeyError:
        return _error(tool_call_id, "read_variable", f"unknown variable: {params.name}")
    if value is None:
        value = ""
    if len(value) > MAX_VARIABLE_VALUE_CHARS:
        # Capture the elided count BEFORE truncation so the marker and the
        # details agree (the RHS is evaluated before rebinding).
        original_len = len(value)
        value = value[:MAX_VARIABLE_VALUE_CHARS] + f"\n[… {original_len} chars total …]"
        shown_len = original_len
    else:
        shown_len = len(value)
    return _text(
        tool_call_id,
        "read_variable",
        value,
        details={"name": params.name, "chars": shown_len},
    )


def build_list_variables_tool() -> AgentTool:
    return AgentTool(
        name="list_variables",
        label="List variables",
        description="List available variable names (never their values).",
        parameters=ListVariablesParams.model_json_schema(),
        approval_tier="read",
        concurrency="shared",
        interruptible=False,
        execute=execute_list_variables,
    )


def build_read_variable_tool() -> AgentTool:
    return AgentTool(
        name="read_variable",
        label="Read variable",
        description="Read the value of one named variable.",
        parameters=ReadVariableParams.model_json_schema(),
        approval_tier="read",
        concurrency="shared",
        interruptible=False,
        execute=execute_read_variable,
    )


# ---------------------------------------------------------------------------
# browser — detect and drive the CMUX browser
# ---------------------------------------------------------------------------

#: Every action the tool accepts. One tuple so the schema text, the dispatch
#: guard and the tests cannot drift apart.
BROWSER_ACTIONS = (
    "open",
    "goto",
    "read",
    "snapshot",
    "screenshot",
    "click",
    "type",
    "close",
    # scroll and logs are served ONLY by the extension bridge; the cmux backend
    # degrades with a typed "use the extension" error rather than faking them
    # (see execute_browser). They are actions so they ride the same schema,
    # approval tier and dispatch as everything else.
    "scroll",
    "logs",
    # tabs lists every live extension-owned tab (all sessions', read-only
    # awareness) so parallel agents can see what is being driven and know which
    # handle to close. Bridge-only like scroll/logs: cmux keeps no multi-surface
    # registry, so it degrades with the same typed "use the extension" error.
    "tabs",
    # The async site-approval flow, extension-only like scroll/logs. open/goto
    # to a not-yet-allowed origin fails EARLY with a typed error naming these
    # two actions, because the old behaviour — blocking the navigation RPC on
    # the popup prompt — expired unseen and read as "bridge unreachable".
    "request_access",
    "await_access",
    "cancel_access",
)

#: ``retitle`` is a wire METHOD but deliberately NOT an action: the SESSION
#: pushes a late-arriving conversation title to the tab group
#: (:func:`retitle_browser_surface`), and the model has no business renaming
#: browser chrome. Advertising it would add schema to every request for a
#: capability no agent should exercise, so this asymmetry between METHODS and
#: BROWSER_ACTIONS is intentional rather than an oversight to be "fixed".

#: Actions that only the Local Operator browser extension can serve. cmux has no
#: console-log tap and no background-tab scroll primitive, so rather than fake a
#: partial result these degrade with a clear, actionable error naming the
#: extension. Kept as a set beside BROWSER_ACTIONS so the degrade check and the
#: advertised action list can never drift apart.
BRIDGE_ONLY_BROWSER_ACTIONS = frozenset(
    {"scroll", "logs", "tabs", "request_access", "await_access", "cancel_access"}
)

#: Direction keywords ``scroll`` accepts. Mirrors extension/src/commands/scroll.ts
#: DIRECTIONS; validated here so a bad keyword is refused before it reaches the
#: wire rather than returning an empty scroll.
_SCROLL_DIRECTIONS = frozenset({"top", "bottom", "up", "down", "left", "right"})

#: Console levels ``logs`` filters on. Mirrors the extension's LEVELS.
_LOG_LEVELS = frozenset({"error", "warning", "info", "log", "all"})

#: Page text is model input, so it rides the same ceiling as command output
#: rather than a bespoke one — a single-page app whose body is megabytes would
#: otherwise spend the whole context window in one tool call.
BROWSER_TEXT_LIMIT_CHARS = BASH_OUTPUT_LIMIT_CHARS

#: How long a navigation gets to prove it actually happened, and how often we
#: ask. Each poll is two cmux round trips (~0.3 s measured on macOS), so the
#: interval is mostly a floor on how hard a slow page hammers the socket.
BROWSER_NAV_TIMEOUT_S = 20.0
BROWSER_NAV_POLL_S = 0.25

#: How long a click gets to START a navigation before we conclude it did not.
#: Measured: cmux's own URL flips within the first poll after a link click, so
#: this is a wide margin. It is also a floor on the latency of a click that
#: does NOT navigate, which is why it is not larger.
BROWSER_CLICK_GRACE_S = 1.5

#: The eight bytes every PNG starts with. Checked because cmux exits 0 after
#: writing a file it never finished painting into.
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

#: One eval per poll instead of three ``get`` calls: each cmux invocation is a
#: process spawn plus a socket round trip (~150 ms measured), and the three
#: values only mean something if they describe the SAME instant.
_DOC_PROBE_JS = "JSON.stringify([document.readyState, location.href, document.title])"

#: Marker property used to notice that a click replaced the document WITHOUT
#: changing the URL — which is exactly what a form POST to the same path does.
#: A fresh document does not carry it, so its disappearance is the signal.
_NAV_TOKEN_SET_JS = "window.__lo_nav = 1; 'ok'"
_NAV_TOKEN_GET_JS = "String(window.__lo_nav || 0)"


def _dom_text_js(selector: str) -> str:
    """Script that extracts text from the DOM without needing layout.

    ``cmux get text`` is ``innerText``, which is defined in terms of RENDERED
    text: a browser surface sitting in a background tab may never lay out, and
    then a page full of content reads as the empty string. Verified on this
    host against a DuckDuckGo results page — both ``get text --selector body``
    and ``document.body.innerText`` returned "" while ``textContent`` held
    15 247 characters.

    script/style/noscript/template are stripped first because ``textContent``
    would otherwise hand the model minified JavaScript as page content, and
    runs of whitespace are collapsed because the un-laid-out DOM keeps every
    source-formatting newline.

    The selector is embedded as a JSON string literal, so a selector
    containing quotes cannot break out of it.
    """
    return (
        "(function(sel){var el=document.querySelector(sel);if(!el)return '';"
        "var c=el.cloneNode(true);"
        "c.querySelectorAll('script,style,noscript,template')"
        ".forEach(function(n){n.remove();});"
        'return c.textContent.replace(/[ \\t\\u00a0]+/g," ")'
        '.replace(/\\n\\s*\\n\\s*\\n+/g,"\\n\\n").trim();})(' + json.dumps(selector) + ")"
    )


#: Schemes the browser may be pointed at. Everything else is refused BEFORE it
#: reaches cmux, because ``cmux browser goto`` is an omnibox: a value it cannot
#: parse as a URL is sent to Google and answered with exit 0 "OK". Verified on
#: this host — ``goto 'not a url at all'`` landed on
#: ``https://www.google.com/search?q=not%20a%20url%20at%20all``, and a
#: ``data:`` URL was search-escaped the same way. Without this guard a typo'd
#: or hallucinated URL yields a search-results page that every later read and
#: screenshot then describes as if it were the requested site.
_BROWSER_URL_SCHEMES = ("http://", "https://")


class BrowserParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str = Field(
        description="open (start a surface at a URL; on the extension backend a "
        "fresh open creates a NEW tab — your session then owns it and reuses it) "
        "| goto | read (page text) | snapshot (accessibility tree with click "
        "refs) | screenshot | click | type | scroll (move the viewport) | logs "
        "(console + errors) | tabs (list all extension-driven tabs, other "
        "sessions' included) | request_access (raise the site-approval prompt "
        "for a not-yet-allowed origin; returns pending/allowed/denied immediately) "
        "| await_access (wait for the user's decision on that prompt) | "
        "cancel_access (cancel YOUR pending exact-origin request) | close (end "
        "YOUR tab when done with it)."
    )
    url: str = Field(
        default="",
        description=(
            "http(s) URL for 'open'/'goto'/'request_access'/'await_access'/" "'cancel_access'."
        ),
    )
    path: str = Field(default="", description="Destination file for 'screenshot'.")
    selector: str = Field(
        default="",
        description="CSS selector or a snapshot ref (e5) for 'click'/'type'; "
        "scopes the text for 'read' (default: body); for 'scroll', the element "
        "to bring into view.",
    )
    text: str = Field(default="", description="Text to enter for 'type'.")
    # scroll params. All optional: with none set, 'scroll' pages one viewport
    # down. Precedence is selector > x/y > direction > default (mirrors
    # extension/src/commands/scroll.ts).
    x: float | None = Field(
        default=None,
        description="'scroll' horizontal pixel delta (positive = right). "
        "Use with 'y' for a precise scrollBy.",
    )
    y: float | None = Field(
        default=None,
        description="'scroll' vertical pixel delta (positive = down).",
    )
    direction: str = Field(
        default="",
        description="'scroll' keyword: top | bottom | up | down | left | right "
        "(top/bottom jump to the extremes; the rest move one page).",
    )
    # logs params.
    level: str = Field(
        default="",
        description="'logs' level filter: error | warning | info | log | all " "(default all).",
    )
    limit: int | None = Field(
        default=None,
        description="'logs' max entries to return (most recent kept); 'scroll' "
        "ignores it. Omit for no cap.",
    )
    timeout_s: float | None = Field(
        default=None,
        description="'await_access' max seconds to wait for the user's decision "
        "(default 120, max 240). Still pending after that? Tell the user, then "
        "call await_access again.",
    )


def _cmux_binary() -> str | None:
    """Absolute path to the cmux CLI, or None when this host has none.

    PATH first, then ``CMUX_BUNDLED_CLI_PATH`` — the variable a cmux session
    exports pointing at the CLI inside the app bundle. That fallback earns its
    place because the bundle's bin directory is prepended to PATH by cmux's
    shell integration, and a venv activation, a ``sudo -i`` or a login shell
    that rebuilds PATH from /etc/paths drops it while every CMUX_* marker
    survives.

    The BINARY is the gate, never the environment markers on their own. CMUX_*
    is inherited by every descendant of a cmux session, including ones that
    crossed into a container or an ssh host where no cmux CLI exists;
    advertising the tool there produced a capability whose every action could
    only answer "cmux is not on PATH". Worth knowing for anyone re-deriving
    this: a real session exports ``CMUX_SOCKET`` EMPTY (the populated variable
    is ``CMUX_SOCKET_PATH``), so the previous ``os.environ.get("CMUX_SOCKET")``
    test could never fire and detection was always really ``which cmux``.
    """
    import shutil

    found = shutil.which("cmux")
    if found:
        return found
    bundled = os.environ.get("CMUX_BUNDLED_CLI_PATH", "").strip()
    if bundled and os.path.isfile(bundled) and os.access(bundled, os.X_OK):
        return bundled
    return None


def cmux_browser_available() -> bool:
    """Whether a CMUX browser can be driven from this session.

    Runs during tool-list construction on every session start, so it stays a
    PATH lookup plus an environment read and spawns nothing. ``cmux
    browser-status`` would additionally report whether the browser panel is
    enabled, but it costs a process spawn and a socket round trip and can hang
    when the socket is wedged — session start must never block on a terminal
    emulator, and a disabled panel already produces a clear per-action error.

    Never raises: an unreadable PATH or environment degrades to "no browser",
    which the createIf builder handles by not advertising the tool at all.

    That degradation is silent to the MODEL by design (advertising a tool
    whose every action errors is worse), which makes it invisible to whoever
    has to explain the absence later. So the one anomalous shape — a session
    carrying cmux's ``CMUX_*`` markers, i.e. plainly running inside cmux, yet
    resolving no CLI — is logged with the markers it saw. A PATH rebuilt by a
    login shell or a ``sudo -i`` is exactly how that happens, and the
    alternative to a log line is diagnosing it from the agent's behaviour
    afterwards. Nothing is logged on an ordinary non-cmux host: absence there
    is normal, and a warning per session start would be noise.
    """
    try:
        if _cmux_binary() is not None:
            return True
        markers = sorted(name for name in os.environ if name.startswith("CMUX_"))
        if markers:
            logger.warning(
                "cmux markers present (%s) but no cmux CLI resolved: not on PATH and "
                "CMUX_BUNDLED_CLI_PATH=%r is not an executable file. The browser tool "
                "will not be advertised this session.",
                ", ".join(markers),
                os.environ.get("CMUX_BUNDLED_CLI_PATH", ""),
            )
        return False
    except Exception:  # noqa: BLE001 — detection must never break session start
        return False


def _browser_state(context: ToolContext | None) -> BrowserSurfaceProtocol:
    """The session's browser surface holder.

    Normally injected by the host (``Session._build_tool_context``), because
    the ToolContext is rebuilt every turn and a handle stored on it would be
    dropped — and a dropped handle strands a cmux tab nothing can close.

    A host that injects nothing still gets a working single call: the
    throwaway holder below means 'open' works and every later action reports
    "no browser surface open".
    """
    if context is None:
        return BrowserSurface()
    if context.browser is None:
        context.browser = BrowserSurface()
    return context.browser


async def _run_cmux(argv: list[str], timeout: float = 30.0) -> tuple[int, str]:
    """Run a cmux subcommand; returns (exit_code, output). Never raises except
    on cancellation, and terminates the child in every exit path so a hung
    cmux cannot orphan.

    The returned text is stdout on success and STDERR on failure, because that
    is where cmux writes its diagnostics — on a socket or permission failure it
    exits non-zero with stdout completely empty. Returning stdout alone made
    every real failure surface to the model and the user as the blank message
    "cmux open failed: ", which is unactionable.
    """
    binary = _cmux_binary()
    if binary is None:
        # Resolved per call rather than cached at import: a session can start
        # before cmux is installed or after PATH is repaired, and a cached
        # "absent" would strand it until restart.
        return 1, "cmux is not on PATH"
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            binary,
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        raw_out, raw_err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        code = proc.returncode or 0
        out = raw_out.decode("utf-8", "replace").strip()
        err = raw_err.decode("utf-8", "replace").strip()
        if code != 0:
            # Prefer stderr, fall back to stdout, and never return "" for a
            # failure — a caller interpolating this must always have something.
            return code, err or out or f"cmux exited {code} with no output"
        return code, out
    except asyncio.CancelledError:
        # BEFORE the handlers below, and re-raised. CancelledError derives from
        # BaseException, so it passes straight through `except
        # asyncio.TimeoutError` and `except Exception` alike, and
        # `wait_for(proc.communicate())` propagates it with the child still
        # running and unreaped. The calls most likely to be cancelled are the
        # long ones — _await_navigation polls up to BROWSER_NAV_TIMEOUT_S and
        # _browser_open allows the full 30 s — which are also the ones most
        # likely to be wedged, so session teardown or an aborted turn would
        # otherwise leave a cmux process behind for each.
        if proc is not None:
            try:
                proc.kill()
                await proc.wait()
            except BaseException:  # noqa: BLE001 — cleanup, incl. a second cancel
                pass
        raise
    except asyncio.TimeoutError:
        if proc is not None:
            try:
                proc.kill()
                await proc.wait()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass
        return 1, f"cmux timed out after {timeout}s"
    except FileNotFoundError:
        return 1, "cmux is not on PATH"
    except Exception as exc:  # noqa: BLE001 — surface, never crash
        return 1, str(exc)


def _cmux_new_surface(url: str) -> list[str]:
    """The open command — and the only sanctioned one.

    ``cmux browser open`` / ``open-split`` / ``new`` reuse a right-hand pane if
    one exists and otherwise SPLIT the user's pane in two, and nothing heals
    that afterwards: the layout is hand-arranged and they rebuild it by hand.
    ``new-surface`` adds the browser as a sibling TAB in the calling pane
    instead. ``--focus false`` keeps their window and workspace exactly where
    they left them; cmux only activates on an explicitly truthy focus.

    No ``--pane``: the socket resolves the calling terminal's own pane, which
    is the pane the browser should join. Omitting ``--workspace`` does NOT
    likewise avoid ``$CMUX_WORKSPACE_ID`` — ``cmux new-surface --help``
    documents ``--workspace <id|ref|index>  Target workspace (default:
    $CMUX_WORKSPACE_ID)``, so cmux applies that default server-side either
    way. It is omitted because passing it explicitly can only make things
    worse (a value we compute from a stale env var, versus cmux resolving its
    own current one), not because it buys immunity from the variable.
    """
    return ["--json", "new-surface", "--type", "browser", "--url", url, "--focus", "false"]


def _surface_argv(surface: str, *rest: str) -> list[str]:
    """``cmux browser --surface <id> ...``.

    The handle always goes in as an OPTION, never as the leading positional
    cmux also accepts, so it can never be re-read as a subcommand.
    """
    return ["browser", "--surface", surface, *rest]


def _validate_browser_url(raw: str, action: str) -> str:
    """Return a refusal message for an unusable URL, or "" when it is fine."""
    url = raw.strip()
    if not url:
        return f"'{action}' requires a URL"
    if url.startswith("-"):
        # The URL also lands in a POSITIONAL argv slot, so a flag-shaped value
        # is parsed by cmux as an option: `goto --help` exits 0 and prints
        # help, which we would then report as a successful navigation. There is
        # no legitimate URL starting with "-".
        return f"refusing a flag-shaped URL: {raw!r}"
    if not url.lower().startswith(_BROWSER_URL_SCHEMES):
        return (
            f"refusing {raw!r}: only http:// and https:// can be opened — cmux "
            "turns anything else into a Google search and still reports success"
        )
    return ""


def _validate_selector(raw: str, action: str) -> str:
    """Return a refusal message for an unusable selector, or "" when fine."""
    selector = raw.strip()
    if not selector:
        return f"'{action}' requires a selector (use 'snapshot' to find one)"
    if selector.startswith("-"):
        # Same argv hazard as the URL: cmux accepts the selector positionally
        # as well as via --selector, and no CSS selector or snapshot ref begins
        # with a dash.
        return f"refusing a flag-shaped selector: {raw!r}"
    return ""


def _validate_typed_text(raw: str) -> str:
    """Return a refusal message for unusable ``type`` text, or "" when fine.

    The same argv hazard the selector and URL are already checked for, in the
    slot that was left out. cmux's parser is flag-greedy in the ``--text``
    position: measured on this host, ``cmux browser --surface <s> fill
    --selector a --text --help`` exits 0 and prints the browser help instead of
    filling anything, and the tool then reported "Typed into a." having typed
    nothing.

    Blanket on a leading dash, matching :func:`_validate_selector`, rather than
    enumerating the values that actually bite. Ordinary dash-leading text
    (``-5``, ``-x``, ``--force``) does fill correctly today, so this refuses a
    little more than it must — but the alternative is an allowlist of cmux's
    global flags, which is a private detail of another program that changes
    without notice. A model that needs a literal leading dash can fill the
    remainder and press the key, and the read-back comparison in
    :func:`_browser_type` catches anything that slips through either way.
    """
    if raw.startswith("-"):
        return (
            f"refusing flag-shaped text: {raw!r} — cmux's --text slot parses a "
            "leading dash as an option (`--text --help` prints help and fills "
            "nothing)"
        )
    return ""


def _validate_browser_args(action: str, params: BrowserParams) -> str:
    """Refuse an unusable argument, or "" when the call may proceed.

    Deliberately one function called from the dispatcher rather than a check at
    the top of each action body, because the ORDER matters and should be
    structural: a value we are going to refuse must not reach the cmux CLI at
    all, not even behind the stale-handle liveness probe. What makes that a
    hard requirement is cmux's own behaviour — ``goto`` is an omnibox that
    Googles a non-URL and still exits 0, and a flag-shaped value in a
    positional or ``--text`` slot is parsed as an option.
    """
    if action in ("open", "goto", "request_access", "await_access", "cancel_access"):
        # The access actions take the SAME url validation as open/goto: they
        # exist to pre-approve exactly the navigation open/goto would make, so
        # a value those verbs would refuse must be refused here too.
        return _validate_browser_url(params.url, action)
    if action in ("click", "type"):
        problem = _validate_selector(params.selector, action)
        if problem or action == "click":
            return problem
        return _validate_typed_text(params.text)
    if action == "snapshot" and params.selector.strip():
        # Optional here — an absent selector snapshots the whole document —
        # which is why this is not the same shape as click/type.
        return _validate_selector(params.selector, "snapshot")
    if action == "scroll":
        # Every scroll param is optional (no params = one viewport down), so the
        # only refusals are a malformed value: a flag-shaped selector (same argv
        # hazard as click) or an unknown direction keyword.
        if params.selector.strip():
            return _validate_selector(params.selector, "scroll")
        direction = params.direction.strip().lower()
        if direction and direction not in _SCROLL_DIRECTIONS:
            return (
                f"unknown scroll direction: {params.direction!r} "
                f"(expected one of {', '.join(sorted(_SCROLL_DIRECTIONS))})"
            )
        return ""
    if action == "logs":
        level = params.level.strip().lower()
        if level and level not in _LOG_LEVELS:
            return (
                f"unknown logs level: {params.level!r} "
                f"(expected one of {', '.join(sorted(_LOG_LEVELS))})"
            )
        return ""
    return ""


def _same_page(live: str, requested: str) -> bool:
    """Whether two URL readings name the same document.

    Only a trailing slash is normalised. Nothing else is: both readings come
    out of the same browser AFTER redirects, so they already agree on scheme,
    host case and query order, and inventing further equivalence would hide
    exactly the mismatch this comparison exists to catch.
    """
    live = live.strip()
    return bool(live) and live.rstrip("/") == requested.strip().rstrip("/")


async def _probe_document(surface: str) -> tuple[tuple[str, str, str] | None, str]:
    """``((readyState, href, title), "")`` or ``(None, diagnostic)``."""
    code, out = await _run_cmux(
        _surface_argv(surface, "eval", "--script", _DOC_PROBE_JS), timeout=15.0
    )
    if code != 0:
        return None, out
    try:
        parsed = json.loads(out.strip())
    except ValueError:
        return None, f"unparseable eval output: {out[:200] or '(empty)'}"
    if not isinstance(parsed, list) or len(parsed) != 3:
        return None, f"unexpected eval payload: {out[:200] or '(empty)'}"
    return (str(parsed[0]), str(parsed[1]), str(parsed[2])), ""


async def _cmux_url_probe(surface: str) -> tuple[int, str]:
    """``(exit_code, url)`` from ``cmux browser --surface <s> get url``.

    Split out because this one verb does double duty: it reports the URL cmux
    is pointing at, AND it is the only cheap liveness check for the handle —
    see :func:`_stale_surface_error`.
    """
    code, out = await _run_cmux(_surface_argv(surface, "get", "url"), timeout=15.0)
    return code, out.strip()


async def _stale_surface_error(
    tool_call_id: str, state: BrowserSurfaceProtocol
) -> ToolResult | None:
    """``None`` when the recorded handle is still live; an error when it is not,
    with the handle cleared.

    Why every surface-taking action pays one extra round trip for this. cmux
    resolves a ``--surface`` handle that no longer exists by falling back to
    the ACTIVE surface and exiting 0 — so a stale handle silently drives
    whatever tab the USER is looking at. Measured on this host against
    ``--surface surface:999999`` (a handle that never existed): ``get title``,
    ``get text --selector body``, ``eval`` and ``snapshot --compact`` all
    returned rc=0 carrying an unrelated tab's content, so ``read`` answered
    ``is_error: False`` with a confident page header and that page's full text
    while ``details.surface_id`` still named the dead handle — internally
    consistent and completely wrong, with nothing in the transcript signalling
    the substitution.

    ``get url`` is the exception that makes the check possible: rc=1,
    ``Error: invalid_params: Missing or invalid surface_id``.
    :func:`_await_navigation` cannot stand in for it, because its ``eval``
    probe SUCCEEDS against the fallback surface and ``probe_failures`` never
    trips.

    Handles go stale routinely: the user closes the tab, or cmux restarts and
    reissues small refs like ``surface:73`` to someone else. Called once per
    action, never inside a poll loop.
    """
    surface = state.surface_id
    code, out = await _cmux_url_probe(surface)
    if code == 0:
        return None
    # Dropped, not kept: a retained dead handle points every later action at
    # the user's own tab, and 'open' reuses the recorded handle, so there would
    # be no route back to a surface of our own.
    state.surface_id = ""
    return _error(
        tool_call_id,
        "browser",
        f"browser surface {surface} is gone ({out or 'no output'}); dropped the "
        "handle. Use 'open' with a URL to get a new surface — acting on this "
        "one would have driven whatever tab is active instead.",
    )


async def _await_navigation(
    surface: str, timeout: float | None = None
) -> tuple[bool, str, str, str]:
    """Block until the live document is the one cmux was asked to load.

    Returns ``(settled, href, title, detail)``; ``detail`` explains the failure
    when ``settled`` is False.

    Why this cannot be skipped. ``cmux browser get url`` answers with the URL
    cmux was last ASKED for, not the URL of the document that is live, and
    ``goto`` exits 0 the instant the request is accepted. Measured on this
    host: after ``goto https://iana.org/domains/example`` — a 301 the WKWebView
    never completed — ``get url`` reported the requested URL for 20+ seconds
    while ``location.href`` and ``get title`` still described the PREVIOUS page
    and ``screenshot`` wrote a byte-identical PNG of it (md5 cef9cd9d…,
    67 821 B). No exit code says so, so without this wait the tool confidently
    reports, reads and photographs the wrong page.

    The two views converge exactly when the requested document is the live one,
    which is what makes their agreement (plus ``readyState``) the completion
    signal. It is redirect-safe because both sides report POST-redirect state —
    verified: ``www.rust-lang.org/learn`` settles with both reading
    ``https://rust-lang.org/learn/``.
    """
    # Read at CALL time, not bound as a default argument: a default freezes the
    # value at import, which would make the module constant unoverridable by a
    # host (or a test) that needs a shorter budget.
    budget = BROWSER_NAV_TIMEOUT_S if timeout is None else timeout
    deadline = time.monotonic() + budget
    href = title = requested = ""
    probe_failures = 0
    while True:
        probe, probe_error = await _probe_document(surface)
        code, out = await _cmux_url_probe(surface)
        if code == 0:
            requested = out
        if probe is None:
            # A single failure is expected mid-navigation: the execution
            # context is destroyed while the new document commits, and eval
            # fails for that instant. A run of them means the surface is gone
            # or the browser panel is disabled, and waiting out the full
            # timeout on that would only delay an error we can already give.
            probe_failures += 1
            if probe_failures >= 3:
                return False, href, title, f"cannot read the document: {probe_error}"
        else:
            probe_failures = 0
            ready, href, title = probe
            if ready == "complete" and _same_page(href, requested):
                return True, href, title, ""
        if time.monotonic() >= deadline:
            return (
                False,
                href,
                title,
                f"after {budget:g}s cmux is pointing at {requested or '(unknown)'} "
                f"but the live document is still {href or '(unreadable)'}",
            )
        await asyncio.sleep(BROWSER_NAV_POLL_S)


_BROWSER_OPEN_CLEANUP_REMINDER = (
    "Close this surface with action='close' before your final response unless the user needs "
    "it left open."
)
_BROWSER_TABS_CLEANUP_FOOTER = (
    "Close YOUR `(yours)` tab when finished; listings cannot be used to close others."
)


def _page_line(title: str, href: str) -> str:
    """One-line description of what is actually on screen.

    Every navigating action ends with this instead of echoing the URL that was
    REQUESTED, so a redirect, a login wall or a consent interstitial shows up
    in the transcript rather than hiding behind the model's own intent.
    """
    return f"{title or '(untitled)'} — {href or '(unknown URL)'}"


async def _browser_goto(tool_call_id: str, surface: str, raw_url: str) -> ToolResult:
    url = raw_url.strip()
    code, out = await _run_cmux(_surface_argv(surface, "goto", url))
    if code != 0:
        return _error(tool_call_id, "browser", f"cmux goto failed: {out}")
    settled, href, title, detail = await _await_navigation(surface)
    if not settled:
        return _error(tool_call_id, "browser", f"navigating to {url} did not complete: {detail}")
    return _text(
        tool_call_id,
        "browser",
        f"Navigated {surface}: {_page_line(title, href)}",
        details={"surface_id": surface, "url": href, "title": title},
    )


async def _browser_open(
    tool_call_id: str, state: BrowserSurfaceProtocol, raw_url: str
) -> ToolResult:
    url = raw_url.strip()
    if state.surface_id:
        live_code, _live_out = await _cmux_url_probe(state.surface_id)
        if live_code == 0:
            # One surface per session, reused. A fresh surface per navigation is
            # how a session leaves a drift of dead browser tabs the user closes
            # one at a time, so 'open' degrades to 'goto' once a surface exists
            # rather than being an error the model has to learn to avoid.
            return await _browser_goto(tool_call_id, state.surface_id, url)
        # The recorded surface is gone (see _stale_surface_error). 'open' is the
        # recovery verb, so it RECOVERS here instead of erroring the way every
        # other action does: drop the dead handle and make a real surface, which
        # is what the model asked for anyway.
        state.surface_id = ""
    code, out = await _run_cmux(_cmux_new_surface(url))
    if code != 0:
        return _error(tool_call_id, "browser", f"cmux open failed: {out.strip()}")
    surface_id = _parse_surface_id(out)
    if not surface_id:
        # A "success" with no handle is a FAILURE: the next goto/screenshot
        # can only report "no browser surface open", and the model has been
        # told the open worked so it has no reason to retry. Fail here, with
        # the raw output, so the actual shape is visible in the transcript.
        return _error(
            tool_call_id,
            "browser",
            "cmux opened a browser but reported no surface handle; cannot "
            f"drive it. Output was: {out or '(empty)'}",
        )
    # Recorded BEFORE the load is confirmed: the surface exists either way, and
    # dropping the handle on a slow page would leak a tab nothing can close.
    state.surface_id = surface_id
    settled, href, title, detail = await _await_navigation(surface_id)
    if not settled:
        return _error(
            tool_call_id,
            "browser",
            f"opened surface {surface_id} but {url} did not load: {detail}. "
            "The surface is open — retry with 'goto', or 'close' it.",
        )
    return _text(
        tool_call_id,
        "browser",
        f"Opened browser surface {surface_id}: {_page_line(title, href)}\n"
        f"{_BROWSER_OPEN_CLEANUP_REMINDER}",
        details={"surface_id": surface_id, "url": href, "title": title},
    )


async def _browser_read(tool_call_id: str, surface: str, raw_selector: str) -> ToolResult:
    # cmux refuses `get text` with no selector, and "body" is the whole page —
    # the default the model means when it says "read the page".
    selector = raw_selector.strip() or "body"
    code, out = await _run_cmux(_surface_argv(surface, "get", "text", "--selector", selector))
    if code != 0:
        return _error(tool_call_id, "browser", f"cmux read failed: {out}")
    if not out.strip():
        # Empty is usually a lie, not an empty page: `get text` is innerText
        # and needs layout the background surface may never have performed.
        # See _dom_text_js — measured 0 characters here against 15 247 in the
        # DOM on a real results page. Falling back beats reporting "(no text)"
        # for a page the model can plainly see in the screenshot.
        fallback_code, fallback_out = await _run_cmux(
            _surface_argv(surface, "eval", "--script", _dom_text_js(selector)), timeout=20.0
        )
        if fallback_code == 0 and fallback_out.strip():
            out = fallback_out
    probe, _probe_error = await _probe_document(surface)
    details: dict[str, Any] = {"surface_id": surface, "selector": selector}
    header = ""
    if probe is not None:
        _ready, href, title = probe
        details["url"] = href
        details["title"] = title
        # Title and URL ride WITH the text. A model that navigated, got
        # redirected to a login wall and then read would otherwise file the
        # content under the URL it asked for.
        header = f"{_page_line(title, href)}\n\n"
    body = truncate_output(out, BROWSER_TEXT_LIMIT_CHARS)
    return _text(tool_call_id, "browser", header + (body or "(no text)"), details=details)


async def _browser_screenshot(
    tool_call_id: str, surface: str, raw_path: str, context: ToolContext | None
) -> ToolResult:
    # cmux takes the destination as ``--out <path>``; passing it positionally
    # is silently IGNORED (cmux writes into its own temp dir and still exits
    # 0), which would make us report a file that does not exist.
    if raw_path:
        # Route through the shared resolver like every other file-writing tool
        # (write/edit/read/grep). Without it `~` was never expanded (creating a
        # literal "~" directory), relative paths resolved against the operator
        # process CWD instead of the session's, and the approval prompt showed
        # the unresolved string the model typed rather than the real target.
        # The approval prompt now shows the RESOLVED destination: the describers
        # in this module run `_display_target`, so a user approving
        # `../../evil.png` reads the absolute path the write will actually touch,
        # with the hazard clause when it leaves the workspace. (The earlier note
        # here described the pre-describer behaviour, where the prompt echoed
        # `call.raw_arguments` and resolution was invisible to the person
        # answering.) `inside` is deliberately unused: unlike read/grep this tool
        # has no read-tier to escalate FROM, and `write` already always prompts.
        resolved, _inside, _resolvable = _resolve_workspace_path(raw_path, _safe_cwd(context))
        target = str(resolved)
    else:
        import tempfile

        target = os.path.join(tempfile.gettempdir(), f"lo-browser-{surface}.png")
    code, out = await _run_cmux(_surface_argv(surface, "screenshot", "--out", target))
    if code != 0:
        return _error(tool_call_id, "browser", f"cmux screenshot failed: {out}")
    # Exit 0 is not proof of a write: confirm the file landed before telling
    # the model it can read it.
    if not os.path.exists(target):
        return _error(
            tool_call_id,
            "browser",
            f"cmux reported success but no file at {target}: {out or '(no output)'}",
        )
    size = os.path.getsize(target)
    with open(target, "rb") as handle:
        magic = handle.read(len(PNG_MAGIC))
    if magic != PNG_MAGIC:
        # A capture of a surface that never painted, or one interrupted
        # mid-write, lands as an empty or truncated file and cmux still exits
        # 0. Catching it here beats the model handing the path to an image
        # reader that fails several turns away from the cause.
        return _error(
            tool_call_id,
            "browser",
            f"{target} is not a PNG ({size} bytes, starts {magic!r}); the "
            "capture did not complete",
        )
    probe, _probe_error = await _probe_document(surface)
    shot_of = ""
    if probe is not None:
        _ready, href, title = probe
        # Says WHAT was photographed, because a screenshot is the one result
        # the model cannot inspect for itself.
        shot_of = f" of {_page_line(title, href)}"
    return _text(
        tool_call_id,
        "browser",
        f"Screenshot{shot_of} saved to {target} ({size} bytes).",
        details={"path": target, "bytes": size},
    )


async def _cmux_url(surface: str) -> str:
    """The URL cmux is POINTING AT — not necessarily the live document's."""
    code, out = await _cmux_url_probe(surface)
    return out if code == 0 else ""


async def _mark_document(surface: str) -> bool:
    """Stamp the live document so a replacement can be noticed. See
    :data:`_NAV_TOKEN_SET_JS`. False when the stamp could not be applied, in
    which case the caller falls back to the URL signal alone."""
    code, _out = await _run_cmux(
        _surface_argv(surface, "eval", "--script", _NAV_TOKEN_SET_JS), timeout=15.0
    )
    return code == 0


async def _navigation_started(surface: str, before: str, marked: bool) -> bool:
    """Whether a click set a navigation in motion.

    A click is asynchronous in a way ``goto`` is not. cmux accepts a ``goto``
    and updates its own URL synchronously, but a link click is initiated by
    the PAGE, so for a short window cmux is still pointing at the old URL and
    both readings agree — which the settle predicate reads as "already
    settled" and reports success on the page we just navigated AWAY from.
    Measured on this host: clicking the example.com link left cmux pointing at
    iana.org/domains/example while the live document stayed example.com for
    20+ seconds, and a settle sampled immediately after the click saw neither
    the departure nor the stall.

    TWO signals, because neither covers the other's case:

    * the URL cmux is pointing at changes — a link click, measured to flip
      within the first poll;
    * the document marker is gone — a form POST to the SAME url, which changes
      no URL at all. Measured against DuckDuckGo's no-JS search form: the
      marker cleared ~0.6 s after submit while the URL never moved, and
      without this signal the result was labelled "no navigation" even though
      the whole document had been replaced.

    An unreadable ``before`` counts as started, so the settle decides —
    guessing "no navigation" there would restore the very bug this prevents.
    """
    if not before:
        return True
    deadline = time.monotonic() + BROWSER_CLICK_GRACE_S
    while True:
        if await _cmux_url(surface) != before:
            return True
        if marked:
            code, out = await _run_cmux(
                _surface_argv(surface, "eval", "--script", _NAV_TOKEN_GET_JS), timeout=15.0
            )
            if code == 0 and out.strip() == "0":
                return True
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(BROWSER_NAV_POLL_S)


async def _browser_click(tool_call_id: str, surface: str, raw_selector: str) -> ToolResult:
    selector = raw_selector.strip()
    before = await _cmux_url(surface)
    marked = await _mark_document(surface)
    code, out = await _run_cmux(_surface_argv(surface, "click", "--selector", selector))
    if code != 0:
        return _error(tool_call_id, "browser", f"cmux click failed: {out}")
    if not await _navigation_started(surface, before, marked):
        # Most clicks do not navigate. Report what is on screen rather than
        # waiting out a load that was never going to happen.
        probe, _probe_error = await _probe_document(surface)
        _ready, href, title = probe if probe is not None else ("", before, "")
        return _text(
            tool_call_id,
            "browser",
            f"Clicked {selector} (no navigation). Page: {_page_line(title, href)}",
            details={"surface_id": surface, "url": href, "title": title},
        )
    settled, href, title, detail = await _await_navigation(surface)
    if not settled:
        return _error(
            tool_call_id,
            "browser",
            f"clicked {selector}, but the page it started loading never "
            f"arrived: {detail}. Anything read now describes the old page.",
        )
    return _text(
        tool_call_id,
        "browser",
        f"Clicked {selector}. Page: {_page_line(title, href)}",
        details={"surface_id": surface, "url": href, "title": title},
    )


async def _browser_type(
    tool_call_id: str, surface: str, raw_selector: str, value: str
) -> ToolResult:
    selector = raw_selector.strip()
    # cmux `fill` REPLACES the field; cmux `type` APPENDS keystrokes to
    # whatever is already there (verified: typing "XY" into a box holding "not
    # a url at all" left "not a url at allXY"). A model that types twice — a
    # retry after a timeout, say — would otherwise silently submit doubled
    # input, and it has no cheap way to notice.
    code, out = await _run_cmux(
        _surface_argv(surface, "fill", "--selector", selector, "--text", value)
    )
    if code != 0:
        return _error(tool_call_id, "browser", f"cmux type failed: {out}")
    read_code, read_out = await _run_cmux(
        _surface_argv(surface, "get", "value", "--selector", selector)
    )
    if read_code != 0:
        # A contenteditable or a non-input target has no `value` property, so
        # an unreadable read-back is not evidence either way and the fill's own
        # exit code stands. Said out loud, because "Typed into X." with no
        # confirmation is otherwise indistinguishable from a verified fill.
        return _text(
            tool_call_id,
            "browser",
            f"Typed into {selector} (cmux could not read the value back, so "
            "this is unverified).",
            details={"surface_id": surface, "selector": selector},
        )
    # Compared, not just echoed. The read-back used to be interpolated straight
    # into "Value is now 'X'." without ever being checked against `value`, so a
    # fill that did nothing was reported as a success quoting the field's OLD
    # contents as the new ones.
    #
    # Whitespace-insensitive on purpose: _run_cmux strips the CLI's output, so
    # leading or trailing spaces in `value` cannot survive the round trip and
    # comparing them would flag every such fill as a failure.
    got = read_out.strip()
    if got != value.strip():
        return _error(
            tool_call_id,
            "browser",
            f"fill of {selector} did not take: the field holds {got!r}, not "
            f"{value!r}. Check the selector names an editable field ('snapshot' "
            "shows what is there).",
        )
    return _text(
        tool_call_id,
        "browser",
        f"Typed into {selector}. Value is now {got!r}.",
        details={"surface_id": surface, "selector": selector},
    )


def bridge_browser_available() -> bool:
    """Cheap browser-bridge discovery, imported lazily off the CLI path."""
    from local_operator.browser_bridge.backend import (
        bridge_browser_available as available,
    )

    return available()


def bridge_browser_advertisable() -> bool:
    """Tool-GATING discovery: cheap, file-only, but honest about a stale daemon.

    Kept distinct from :func:`bridge_browser_available` because gating asks a
    weaker question than backend selection does; see
    :func:`local_operator.browser_bridge.state.advertisable`.
    """
    from local_operator.browser_bridge.backend import (
        bridge_browser_advertisable as advertisable,
    )

    return advertisable()


async def bridge_browser_reachable(classified: tuple[Any, Any] | None = None) -> bool:
    """Browser-path availability: the file probe, plus one socket confirmation.

    Used instead of :func:`bridge_browser_available` everywhere a browser
    ACTION decides which backend to use, because the cheap file probe reports a
    healthy daemon as gone once its heartbeat writer stops. It only pays for a
    socket when it is about to condemn a bridge whose pid is alive.

    The cheap check runs FIRST and short-circuits, so the common case costs
    exactly what it did before and no socket is opened. The probe exists only
    to ACQUIT a daemon the file was about to condemn. ``classified`` is a
    reading the caller already took; it is handed to the probe so that a
    condemning action classifies the daemon once instead of once per consumer.
    """
    if bridge_browser_available():
        return True
    from local_operator.browser_bridge.backend import (
        bridge_browser_reachable as reachable,
    )

    return await reachable(classified=classified)


def _bridge_liveness() -> tuple[Any, Any]:
    """Classify the daemon from the file, never raising at a diagnostic site."""
    from local_operator.browser_bridge import state as state_store

    try:
        return state_store.liveness()
    except Exception:  # noqa: BLE001 - a diagnostic may never raise
        return None, None


def _bridge_demotion_hint(classified: tuple[Any, Any] | None = None) -> str:
    """Why the extension is not being used, when it looked like it should be.

    A session that had been driving the extension and then finds it
    unavailable used to get a bare "not supported on the cmux backend", with
    nothing linking that to the bridge having gone away. That is what led an
    agent to conclude the bridge was "bound" by an orphan tab and abandon it
    for a whole session. Naming the demotion and the one-line repair turns an
    hour of guessing into a single command.

    ``classified`` is the ``(status, state)`` the caller already computed. Each
    demotion path had re-read the discovery file here, which cost a second read
    per demoted action and, worse, could disagree with the classification that
    caused the demotion: a heartbeat refreshed in between made this return ""
    and silently dropped the diagnostic in exactly the race where the daemon
    had just recovered. Reusing the caller's answer keeps the message and the
    decision consistent by construction.
    """
    from local_operator.browser_bridge import state as state_store

    status, current = classified if classified is not None else _bridge_liveness()
    if status is state_store.Liveness.STALE and current is not None:
        age = state_store.heartbeat_age(current)
        return (
            f" NOTE: this session was DEMOTED from the extension to cmux — the bridge daemon "
            f"(pid {current.pid}) is running but its discovery heartbeat is {age:.0f}s stale, "
            "so it advertised itself as unavailable. Run 'lop browser status --repair' to "
            "reconcile it, then retry."
        )
    return ""


def _browser_requester(context: ToolContext | None, tool_call_id: str) -> str:
    """The session-scoped identity the extension binds approvals to.

    A once-grant earned by request_access must be spendable by the SAME
    session's next open/goto but not by a parallel session's; per-command
    request ids cannot express that (every command mints a fresh one), so the
    identity is the session id. A host without one falls back to the tool
    call id — still bound to SOMETHING rather than anonymous, and anonymous
    would be a fail-open hole in the cross-session grant check."""
    if context is not None and context.session_id:
        return f"session:{context.session_id}"
    return f"call:{tool_call_id}"


_BROWSER_SESSION_LABEL_CLUSTERS = 30

#: Last-resort label for a session with neither a title nor a usable cwd. A
#: bare ``Session`` on every group at once is what the ordinal de-duplication
#: then turns into ``LO · Session (2)``/``(3)`` — technically distinct, and
#: naming nothing. It survives only for a session running from a filesystem
#: root, where ``cwd_label`` itself declines to answer.
_BROWSER_FALLBACK_LABEL = "Session"

#: Separator between a subagent's parent conversation and its own job label.
#: U+203A, matching the TUI's own ``lo › <cwd>`` composition rather than
#: introducing a second convention for the same "A, narrowed to B" relation.
_BROWSER_SUBAGENT_SEPARATOR = " › "

#: Below this many clusters a clipped parent title identifies no conversation
#: (``Fi…`` names nothing), so the child's label stands alone instead of
#: wearing a stub prefix. See :func:`_browser_subagent_label`.
_BROWSER_SUBAGENT_PARENT_MIN = 8


def _browser_live_session_name(context: ToolContext | None) -> str:
    """The session's title as of NOW, not as of the turn's context snapshot.

    ``ToolContext`` is rebuilt once per turn, while a conversation names itself
    a second or two INTO its first turn (asynchronously, alongside the turn).
    A browse in that opening turn — the common case, since "look at this page"
    is often the opening request — therefore read an empty ``session_name``
    even after the title had landed. The provider re-reads the session's live
    holder; the snapshot remains the fallback for hosts that install no
    provider. Display-only, and guarded: a host callback must never be able to
    fail a browse.
    """
    if context is None:
        return ""
    provider = context.session_name_provider
    if provider is not None:
        try:
            live = provider()
        except Exception:  # noqa: BLE001 — a label is never worth a failed browse
            logger.debug("browser: live session-name provider failed", exc_info=True)
        else:
            if isinstance(live, str) and live.strip():
                return live
    return context.session_name


def _browser_clean_label(raw: str) -> str:
    """``raw`` with control/format characters removed and whitespace collapsed.

    Removing all control/format characters is intentionally broader than the
    known bidi and zero-width set: browser tab chrome is too small to make
    invisible direction changes attributable. Returns ``""`` for anything that
    sanitises away to nothing, which is what lets callers treat "invisible" and
    "absent" as the same state rather than shipping a title the user cannot see.
    """
    cleaned = "".join(
        " " if char.isspace() else char
        for char in raw
        if unicodedata.category(char) not in {"Cf"}
        and not (unicodedata.category(char) == "Cc" and not char.isspace())
    )
    return " ".join(cleaned.split())


def _browser_session_label(context: ToolContext | None) -> str:
    """Display-only session title safe for compact browser chrome.

    The extension receives identity separately, so this value may never fall
    back to a UUID or request token. Removing all control/format characters is
    intentionally broader than the known bidi and zero-width set: browser tab
    chrome is too small to make invisible direction changes attributable.

    A SUBAGENT is named first, before any title lookup, because it is the one
    session kind that can never acquire a title: naming lives in the TUI host
    and the owned-session runtime, and a one-shot child runs through neither
    (confirmed on this machine — no subagent session directory holds a
    ``title.json``). Its context carries no ``session_name`` and its cwd is its
    PARENT's, so every child of one parent derived the identical cwd label and
    a fleet of them rendered as ``LO · local-operator (2)``/``(3)``/… — the
    ordinal is the only thing that differed, and it names nothing. See
    :func:`_browser_subagent_label` for the form and why the parent's name is
    carried alongside the child's.

    An UNNAMED top-level session falls back to its working directory's
    basename, which is the same substitution the TUI's status band and terminal
    title already make in this exact slot (``tui/terminal_title.cwd_label`` —
    ``lo › <cwd>``). The slot has always held "the best label we have", and a
    directory the user chose the session for distinguishes three concurrent
    groups where three copies of ``Session`` do not.
    """
    # Subagent first: a child's own ``session_name`` is empty by construction,
    # but the cwd fallback below would still answer (with the parent's
    # directory) and thereby hide the far more specific label the parent
    # launched this child under. Order is the whole fix.
    subagent = _browser_clean_label(_browser_subagent_label(context))
    if subagent:
        return _browser_clip_label(subagent)
    # Emptiness is decided by the SANITISED text, not by ``str.strip()``.
    # ``strip()`` removes whitespace but NOT the Cf/Cc classes, so a title of
    # nothing but zero-width or bidi characters (U+200B, U+FEFF, U+2060,
    # U+200E, \x01) read as a real name, survived to the sanitiser, emptied
    # there, and landed on the bare fallback — skipping the cwd substitution
    # this function exists to make. Sanitising first is what makes "is there a
    # usable name here?" and "what will the user actually see?" the same
    # question (QA round 1, Q2).
    cleaned = _browser_clean_label(_browser_live_session_name(context))
    if not cleaned and context is not None:
        cleaned = _browser_clean_label(_browser_cwd_label(context.cwd))
    if not cleaned:
        return _BROWSER_FALLBACK_LABEL
    return _browser_clip_label(cleaned)


def _browser_clusters(text: str) -> list[str]:
    """``text`` split into grapheme-ish clusters (base char + its marks).

    The pill's budget is counted in what the user perceives as characters, so
    combining marks and modifier symbols ride with the base they attach to
    rather than each costing a slot. Approximates ``Intl.Segmenter``, which is
    what the extension side counts with; exactness is not required because
    both sides only ever clip, never realign.
    """
    clusters: list[str] = []
    for char in text:
        if clusters and (unicodedata.combining(char) or unicodedata.category(char) == "Sk"):
            clusters[-1] += char
        else:
            clusters.append(char)
    return clusters


def _browser_clip_label(cleaned: str, budget: int = _BROWSER_SESSION_LABEL_CLUSTERS) -> str:
    """``cleaned`` cut to ``budget`` grapheme clusters, ellipsised if cut.

    Split out of :func:`_browser_session_label` so the subagent form goes
    through the SAME clip as a conversation title: both are user-visible text
    of unbounded length landing in the same 30-cluster pill, and a second
    hand-rolled truncation beside this one is how the two drift apart. Expects
    text that has already been through :func:`_browser_clean_label`.

    ``budget`` is inclusive of the ellipsis, so the RESULT never exceeds it.
    It previously counted the clusters KEPT and then appended the ellipsis on
    top, which returned 31 clusters against a documented 30 — harmless in
    practice, since the extension's ``cleanLabel`` re-clips anything it is sent
    (its ``slice(0, MAX) + "…"`` has the same shape), but it made every
    statement of the ceiling in this module false by one and left the composed
    subagent form paying for the ellipsis with a hand-rolled ``budget - 1``.
    A pill one cluster shorter is the price of a bound that is actually true;
    the extension now never has to re-clip a label this side produced.

    ``budget`` is a parameter only because the subagent composition spends part
    of the pill on the child's label and the separator; every other caller
    takes the full width.
    """
    clusters = _browser_clusters(cleaned)
    if len(clusters) <= budget:
        return cleaned

    clipped = "".join(clusters[: max(budget - 1, 0)]).rstrip()
    # Prefer a complete word when that still leaves a useful title; long words
    # fall back to the grapheme-safe hard boundary rather than an empty label.
    word_boundary = clipped.rfind(" ")
    if word_boundary >= 8:
        clipped = clipped[:word_boundary].rstrip()
    return f"{clipped}…" if clipped else _BROWSER_FALLBACK_LABEL


def _browser_subagent_label(context: ToolContext | None) -> str:
    """``<parent title> › <job label>`` for a subagent, else ``""``.

    ``job_id`` is the discriminator, not ``job_label``: the TUI host leaves it
    unset, so the operator's own session can never be mistaken for a child. It
    is NOT unique to subagents — a server-side session carries one too (see
    ``server/utils/operator.py`` and the queued-job path) — which is why the
    two fields are required TOGETHER below rather than ``job_id`` alone: only
    ``_build_child_session`` sets ``job_label``, so a server session reaches
    the ``label or parent`` branch and keeps its own title unchanged. Stated
    because a future reader deciding what may set ``job_id`` would otherwise
    rely on an exclusivity that does not hold.

    A child whose label is missing or sanitises away still gets the parent's
    name rather than falling through to the shared-cwd label every sibling
    would also derive.

    BOTH halves are carried because each answers a different question the
    operator actually asks of a tab group. The child's label says WHICH slice
    of work this is (it is what they typed into ``task`` and what the jobs list
    and ``hub`` address it by); the parent's title says WHICH CONVERSATION
    spawned it, which is the part that separates two concurrent sessions both
    running a child called ``qa``. The separator is U+203A, matching the TUI's
    own ``lo › <cwd>`` composition rather than inventing a second convention.

    The parent's name is read through the normal live/snapshot chain, so a
    parent titled after this child was launched still reaches the pill on the
    child's next command.

    When the composition does not fit, the PARENT half is what gets clipped —
    the child's label is kept whole. Clipping the composed string as one unit
    (the obvious implementation) is wrong here and was measured to be: a real
    pair, ``Fix Slack-reported UI zoom and overlap bugs`` + ``zoom-scroll-fix``,
    clipped to ``Fix Slack-reported UI zoom…`` and lost the label entirely,
    leaving every sibling of that parent identical again — precisely the
    failure this function exists to fix. The label is the distinguishing half
    (siblings share a parent by definition), so it holds its ground and the
    shared prefix absorbs the loss.

    Past a point there is nothing left to absorb it with, and the parent half
    DISAPPEARS rather than shrinking to a stub: with a 30-cluster pill, a
    3-cluster separator and an 8-cluster minimum for a parent worth reading,
    that happens once the child's own label reaches 20 clusters. Beyond that
    the pill is the bare label — correct (the label is what distinguishes
    siblings) but worth knowing when a fleet of long-labelled children shows
    no conversation prefix at all.
    """
    if context is None or context.job_id is None:
        return ""
    label = _browser_clean_label(context.job_label)
    # The parent's title, never the child's: a child has no ``session_name`` of
    # its own, and the provider chain is what picks up a parent renamed since
    # launch. Falls back to the parent's cwd basename for an unnamed parent —
    # the same substitution a top-level session gets — so the composed form
    # degrades to `<dir> › <label>` rather than to a bare label.
    parent = _browser_clean_label(_browser_live_session_name(context))
    if not parent and context.cwd:
        parent = _browser_clean_label(_browser_cwd_label(context.cwd))
    if not (parent and label):
        # Only one half is available, so the caller's full-width clip applies
        # to it unchanged.
        return label or parent
    # Budget in CLUSTERS, matching what the clip and the extension count, and
    # measure the composed result rather than assuming it fits: a parent that
    # is already short enough must not be ellipsised for nothing.
    room = (
        _BROWSER_SESSION_LABEL_CLUSTERS
        - len(_browser_clusters(label))
        - len(_browser_clusters(_BROWSER_SUBAGENT_SEPARATOR))
    )
    # A parent squeezed below ``_BROWSER_SUBAGENT_PARENT_MIN`` is not worth
    # showing — ``Fi…`` identifies no conversation and merely steals room from
    # the half that does identify something — so the label stands alone. This
    # also covers a label long enough to fill the pill by itself, where ``room``
    # goes negative; the caller then clips the label on the normal path.
    if room < _BROWSER_SUBAGENT_PARENT_MIN:
        return label
    # Guarded rather than clipped unconditionally: ``_browser_clip_label``
    # returns its input untouched when it already fits, so this only spells out
    # that a parent short enough to fit is never ellipsised for nothing.
    if len(_browser_clusters(parent)) > room:
        parent = _browser_clip_label(parent, room)
    return f"{parent}{_BROWSER_SUBAGENT_SEPARATOR}{label}"


def _browser_cwd_label(cwd: str) -> str:
    """The working directory's basename, or ``""`` at a filesystem root.

    Deliberately re-derived here rather than imported from
    ``tui.terminal_title``: the tool layer must not depend on the TUI (headless
    hosts — server, exec, mobile — build browser params too, and the TUI module
    pulls in Textual settings). The RULE is shared, not the code, and it is one
    line; ``cwd_label``'s root case is mirrored because ``LO · /`` names a
    session no better than ``LO · Session`` does. Sanitisation is the caller's:
    this feeds the same cleaner every other label goes through.
    """
    if not cwd:
        return ""
    path = Path(cwd)
    return "" if path.name in ("", path.anchor) else path.name


def _browser_identity_params(context: ToolContext | None, tool_call_id: str) -> dict[str, str]:
    """Trusted wire metadata; model/browser arguments cannot override it."""
    return {
        "requester": _browser_requester(context, tool_call_id),
        "session_label": _browser_session_label(context),
    }


async def _bridge_call(
    tool_call_id: str,
    action: str,
    params: dict[str, Any],
    *,
    surface: str = "",
) -> tuple[dict[str, Any] | None, ToolResult | None]:
    from local_operator.browser_bridge.backend import (
        BridgeClient,
        BridgeError,
        BridgeUnreachable,
        format_error,
    )

    try:
        return await BridgeClient().call(action, params), None
    except BridgeError as exc:
        problem = _error(
            tool_call_id,
            "browser",
            format_error(exc, action=action, surface=surface),
        )
        # Carry the TYPED wire code so callers can branch on it (dead-pin
        # recovery, handle-drop) instead of substring-matching the human
        # diagnostic, which broke silently on any rewording (finding m4).
        problem.details = {"error_code": exc.code.value}
        return None, problem
    except BridgeUnreachable as exc:
        return None, _error(tool_call_id, "browser", str(exc))


async def _bridge_open(
    tool_call_id: str,
    state: BrowserSurfaceProtocol,
    raw_url: str,
    context: ToolContext | None = None,
) -> ToolResult:
    # The session's pinned handle decides the mode. With one, `open` RESUMES
    # that tab (extension-side it navigates exactly that surface); without one
    # the extension creates a brand-new tab. The extension never falls back to
    # reusing some other live surface — that reuse is how one session used to
    # hijack another's tab mid-task when agents ran in parallel.
    params: dict[str, Any] = {
        "url": raw_url.strip(),
        **_browser_identity_params(context, tool_call_id),
    }
    resuming = state.surface_id.startswith("bridge:")
    created_new = not resuming
    if resuming:
        params["tab"] = state.surface_id
    result, problem = await _bridge_call(tool_call_id, "open", params, surface=state.surface_id)
    if (
        problem is not None
        and resuming
        and (problem.details or {}).get("error_code") == "tab_closed"
    ):
        # The pinned tab died (user closed it, browser restarted). 'open' is
        # the recovery verb — same contract as the cmux path — so drop the
        # dead handle and create a fresh tab instead of surfacing the error.
        state.surface_id = ""
        created_new = True
        result, problem = await _bridge_call(
            tool_call_id,
            "open",
            {"url": raw_url.strip(), **_browser_identity_params(context, tool_call_id)},
        )
    if problem is not None:
        return problem
    assert result is not None
    surface = str(result.get("tab", ""))
    if not re.match(r"^bridge:\d+:[A-Za-z0-9_-]+$", surface):
        return _error(
            tool_call_id, "browser", "browser extension opened a tab but returned no valid handle"
        )
    state.surface_id = surface
    href = str(result.get("url", ""))
    title = str(result.get("title", ""))
    message = f"Opened browser surface {surface}: {_page_line(title, href)}"
    if created_new:
        message += f"\n{_BROWSER_OPEN_CLEANUP_REMINDER}"
    return _text(
        tool_call_id,
        "browser",
        message,
        details={"surface_id": surface, "url": href, "title": title},
    )


#: await_access defaults and cap. The cap exists because each slice is a real
#: RPC and the human may simply be away: 240 s is long enough for "walk back to
#: the desk", short enough that the agent gets a turn to re-notify the user
#: rather than sitting silent for the extension's whole 10-minute request TTL.
BROWSER_AWAIT_ACCESS_DEFAULT_S = 120.0
BROWSER_AWAIT_ACCESS_MAX_S = 240.0

#: One extension-side wait slice (mirrors access.ts AWAIT_SLICE_MS). The tool
#: loops short slices instead of asking the daemon for one long wait so every
#: entry in the daemon's COMMAND_TIMEOUTS stays an honest per-RPC bound — the
#: deadline-extension special case is exactly what made the old blocking flow's
#: failures unreadable.
_BRIDGE_AWAIT_SLICE_MS = 20_000


def _access_result_text(
    state: str, origin: str, *, position: int | None = None, pending_count: int | None = None
) -> str:
    """One agent-facing line per access state, including the next step — the
    agent discovers this flow through error/result text, not documentation."""
    if state == "allowed":
        return f"{origin} is allowed. 'open' or 'goto' the URL now."
    if state == "denied":
        return (
            f"the user denied access to {origin}. Do not retry or re-request this "
            "origin; ask the user directly if it is essential."
        )
    if state == "pending":
        # NOTIFY-FIRST is load-bearing: Chrome's own notification banner is
        # best-effort (macOS suppresses it without Notification Center
        # authorization), so if the agent does not message the user the prompt
        # sits unseen until its TTL — the exact incident this flow replaces.
        return (
            f"approval for {origin} is pending"
            + (f" ({position} of {pending_count})" if position and pending_count else "")
            + ". FIRST notify the user (via the ask "
            "tool or a message) to approve it in the Local Operator extension popup "
            "(toolbar icon, numbered badge showing the pending count) — the badge alone "
            "is not reliably seen — THEN "
            "call action='await_access' with the same url to wait for the decision."
        )
    if state == "superseded":
        # A DIFFERENT session's request replaced this one's prompt slot (the
        # popup shows one origin at a time). The agent must know it was
        # displaced — reading this as expiry would send it into a
        # request/notify loop that keeps stealing the prompt back and forth
        # between sessions (round-1 B1b).
        return (
            f"the approval prompt for {origin} was superseded by another session's "
            "request — the extension shows one prompt at a time. Wait for the other "
            "session's prompt to resolve, then call action='request_access' again "
            "if this origin is still needed."
        )
    if state == "cancelled":
        return f"your pending access request for {origin} was cancelled."
    # "none": no live request for the caller — expired or never raised.
    return (
        f"no live access request for {origin} (it may have expired unanswered, or "
        "never been raised). Call action='request_access' with the url to raise a "
        "new prompt."
    )


async def _bridge_access(
    tool_call_id: str,
    action: str,
    params: BrowserParams,
    context: ToolContext | None,
) -> ToolResult:
    """request_access / await_access / cancel_access — surface-free by design: they exist for
    the moment when 'open' has just FAILED, so requiring an open surface here
    would deadlock the recovery path."""
    url = params.url.strip()
    identity = _browser_identity_params(context, tool_call_id)
    if action == "request_access":
        result, problem = await _bridge_call(
            tool_call_id, "request_access", {"url": url, **identity}
        )
        if problem is not None:
            return problem
        assert result is not None
        state_value = str(result.get("state", ""))
        origin = str(result.get("origin", url))
        return _text(
            tool_call_id,
            "browser",
            _access_result_text(
                state_value,
                origin,
                position=result.get("position"),
                pending_count=result.get("pending_count"),
            ),
            details={
                "origin": origin,
                "state": state_value,
                **{
                    key: result[key]
                    for key in ("position", "pending_count", "expires_at")
                    if key in result
                },
            },
        )
    if action == "cancel_access":
        result, problem = await _bridge_call(
            tool_call_id, "cancel_access", {"url": url, **identity}
        )
        if problem is not None:
            return problem
        assert result is not None
        state_value = str(result.get("state", "none"))
        origin = str(result.get("origin", url))
        return _text(
            tool_call_id,
            "browser",
            _access_result_text(state_value, origin),
            details={
                "origin": origin,
                "state": state_value,
                **({"pending_count": result["pending_count"]} if "pending_count" in result else {}),
            },
        )
    # await_access: loop bounded extension-side slices until the decision, the
    # caller's deadline, or a terminal state. Each slice is its own RPC well
    # inside the daemon's command timeout, so a slow human can never make the
    # bridge look broken again.
    budget = params.timeout_s if params.timeout_s and params.timeout_s > 0 else None
    total_s = min(budget or BROWSER_AWAIT_ACCESS_DEFAULT_S, BROWSER_AWAIT_ACCESS_MAX_S)
    deadline = time.monotonic() + total_s
    while True:
        remaining_ms = int((deadline - time.monotonic()) * 1000)
        if remaining_ms <= 0:
            return _text(
                tool_call_id,
                "browser",
                f"still pending after {total_s:.0f}s: the user has not decided on {url} "
                "yet. Remind them to check the Local Operator extension popup, then call "
                "await_access again.",
                details={"origin": url, "state": "pending"},
            )
        wire = {
            "url": url,
            "timeout_ms": min(remaining_ms, _BRIDGE_AWAIT_SLICE_MS),
            **identity,
        }
        result, problem = await _bridge_call(tool_call_id, "await_access", wire)
        if problem is not None:
            return problem
        assert result is not None
        state_value = str(result.get("state", ""))
        if state_value != "pending":
            origin = str(result.get("origin", url))
            return _text(
                tool_call_id,
                "browser",
                _access_result_text(
                    state_value,
                    origin,
                    position=result.get("position"),
                    pending_count=result.get("pending_count"),
                ),
                details={
                    "origin": origin,
                    "state": state_value,
                    **{
                        key: result[key]
                        for key in ("position", "pending_count", "expires_at")
                        if key in result
                    },
                },
            )


def _format_log_entry(entry: dict[str, Any]) -> str:
    """One buffered log line rendered for the model: ``[level] text (url:line)``.

    The source (console vs exception) is folded into the level tag so an
    uncaught exception is visually distinct from a plain ``console.error`` — the
    difference that matters when the agent is debugging a broken page.
    """
    level = str(entry.get("level", "log")).upper()
    source = str(entry.get("source", ""))
    tag = f"{level}!" if source == "exception" else level
    text = str(entry.get("text", "")).strip()
    url = str(entry.get("url", "")).strip()
    line = entry.get("line")
    where = ""
    if url:
        where = f" ({url}:{line})" if line else f" ({url})"
    return f"[{tag}] {text}{where}"


def _bridge_logs_result(
    tool_call_id: str,
    params: BrowserParams,
    result: dict[str, Any],
    details: dict[str, Any],
    title: str,
    href: str,
) -> ToolResult:
    """Render the extension's buffered log entries, newest-last.

    The joined text rides the same BROWSER_TEXT_LIMIT_CHARS ceiling as read and
    snapshot: a page in a console-spam loop must not spend the whole context
    window in one call, and the extension already caps its ring buffer, so this
    is the second, model-facing cap.
    """
    entries = result.get("entries")
    if not isinstance(entries, list) or not entries:
        level = params.level.strip().lower() or "all"
        scope = "" if level == "all" else f" at level '{level}'"
        return _text(
            tool_call_id,
            "browser",
            f"No console logs{scope} since the page opened. Page: {_page_line(title, href)}",
            details={**details, "log_count": 0},
        )
    lines = [_format_log_entry(entry) for entry in entries if isinstance(entry, dict)]
    body = truncate_output("\n".join(lines), BROWSER_TEXT_LIMIT_CHARS)
    return _text(
        tool_call_id,
        "browser",
        f"{len(lines)} log entr{'y' if len(lines) == 1 else 'ies'} " f"(newest last):\n\n{body}",
        details={**details, "log_count": len(lines)},
    )


def _owns_redacted_tab(own_surface: str, redacted: str) -> bool:
    """Whether the session's full pinned handle names a REDACTED listing entry.

    The extension truncates listed nonces (`bridge:<tabId>:<prefix>…`) so a
    listing cannot hand out drive capabilities (review finding M1); a session
    recognises its own tab by prefix-matching the full token it was given at
    open. Mirrors ownsRedacted in extension/src/state.ts.
    """
    if not own_surface or not redacted:
        return False
    if redacted.endswith("…"):
        return own_surface.startswith(redacted[:-1])
    return own_surface == redacted


def _format_bridge_tab(entry: dict[str, Any], own_surface: str) -> str:
    """One listed tab: redacted handle, page, recency — the caller's own marked.

    The "(yours)" marker matters because the listing shows EVERY session's
    tabs: an agent must close its own when done, and must treat the rest as
    read-only awareness. The handles are redacted by the extension and are NOT
    driveable — driving needs the full token 'open' returned to its owner.
    """
    token = str(entry.get("tab", ""))
    title = str(entry.get("title", "")).strip() or "(untitled)"
    url = str(entry.get("url", "")).strip() or "(no URL)"
    mine = " (yours)" if _owns_redacted_tab(own_surface, token) else ""
    when = ""
    last_used = entry.get("lastUsedAt")
    if isinstance(last_used, (int, float)) and last_used > 0:
        stamp = datetime.fromtimestamp(last_used / 1000, tz=UTC)
        when = f" — last used {stamp.strftime('%H:%M:%S')} UTC"
    return f"{token}{mine}: {title} — {url}{when}"


async def _bridge_tabs(tool_call_id: str, state: BrowserSurfaceProtocol) -> ToolResult:
    """List every live extension-driven tab (all sessions').

    Discovery deliberately needs no open surface of our own: its main use is a
    session deciding whether to resume, or being told the surface cap is hit
    and needing to see what is already open. The extension prunes dead tabs as
    part of answering, so the list is live by construction.
    """
    result, problem = await _bridge_call(tool_call_id, "tabs", {}, surface=state.surface_id)
    if problem is not None:
        return problem
    assert result is not None
    entries = [entry for entry in result.get("tabs") or [] if isinstance(entry, dict)]
    if not entries:
        return _text(
            tool_call_id,
            "browser",
            "No extension-driven browser tabs are open. Use 'open' with a URL to start one.\n\n"
            f"{_BROWSER_TABS_CLEANUP_FOOTER}",
            details={"tab_count": 0},
        )
    lines = [_format_bridge_tab(entry, state.surface_id) for entry in entries]
    return _text(
        tool_call_id,
        "browser",
        f"{len(entries)} extension-driven tab{'s' if len(entries) != 1 else ''} "
        "(most recently used first; handles are redacted — the listing is "
        "awareness-only and cannot drive or close a tab. Your own tab is "
        "marked '(yours)'; drive it with the handle your session already "
        "holds):\n\n" + "\n".join(lines) + f"\n\n{_BROWSER_TABS_CLEANUP_FOOTER}",
        details={"tab_count": len(entries), "surface_id": state.surface_id},
    )


async def _bridge_action(
    tool_call_id: str,
    state: BrowserSurfaceProtocol,
    action: str,
    params: BrowserParams,
    context: ToolContext | None,
) -> ToolResult:
    surface = state.surface_id
    wire: dict[str, Any] = {
        "tab": surface,
        # Identity and display label are trusted host metadata. Keeping both on
        # every owned-tab command lets renames propagate without changing the
        # stable requester used by access receipts and one-shot grants.
        **_browser_identity_params(context, tool_call_id),
    }
    if action == "goto":
        wire["url"] = params.url.strip()
    elif action in ("read", "snapshot"):
        if params.selector.strip():
            wire["selector"] = params.selector.strip()
    elif action in ("click", "type"):
        wire["selector"] = params.selector.strip()
        if action == "type":
            wire["text"] = params.text
    elif action == "scroll":
        # Only send the params the caller actually set, so the extension's
        # precedence (selector > x/y > direction > default) sees an unset param
        # as absent rather than an empty-string selector or a zero delta.
        if params.selector.strip():
            wire["selector"] = params.selector.strip()
        if params.x is not None:
            wire["x"] = params.x
        if params.y is not None:
            wire["y"] = params.y
        if params.direction.strip():
            wire["direction"] = params.direction.strip().lower()
    elif action == "logs":
        wire["level"] = params.level.strip().lower() or "all"
        if params.limit is not None:
            wire["limit"] = params.limit
    result, problem = await _bridge_call(tool_call_id, action, wire, surface=surface)
    if problem is not None:
        # A nonce-invalid or user-closed tab must be forgotten immediately;
        # retaining it would make even the recovery verb target stale state.
        # Branch on the typed code, not the diagnostic's wording (finding m4).
        if (problem.details or {}).get("error_code") == "tab_closed":
            state.surface_id = ""
        return problem
    assert result is not None
    href = str(result.get("url", ""))
    title = str(result.get("title", ""))
    details: dict[str, Any] = {"surface_id": surface}
    if href:
        details.update(url=href, title=title)
    if action == "goto":
        return _text(
            tool_call_id,
            "browser",
            f"Navigated {surface}: {_page_line(title, href)}",
            details=details,
        )
    if action == "read":
        selector = params.selector.strip() or "body"
        details["selector"] = selector
        body = truncate_output(str(result.get("text", "")), BROWSER_TEXT_LIMIT_CHARS)
        return _text(
            tool_call_id,
            "browser",
            f"{_page_line(title, href)}\n\n{body or '(no text)'}",
            details=details,
        )
    if action == "snapshot":
        return _text(
            tool_call_id,
            "browser",
            truncate_output(str(result.get("snapshot", "")), BROWSER_TEXT_LIMIT_CHARS)
            or "(empty snapshot)",
            details=details,
        )
    if action == "scroll":
        x = int(result.get("scrollX", 0) or 0)
        y = int(result.get("scrollY", 0) or 0)
        more_below = bool(result.get("moreBelow"))
        more_right = bool(result.get("moreRight"))
        # Tell the agent where it landed AND whether paging further will reveal
        # more, so it can stop at the end instead of scrolling into a wall.
        remaining = []
        if more_below:
            remaining.append("more below")
        if more_right:
            remaining.append("more right")
        edge = ", ".join(remaining) if remaining else "at the end (no more content)"
        details.update(scroll_x=x, scroll_y=y, more_below=more_below, more_right=more_right)
        return _text(
            tool_call_id,
            "browser",
            f"Scrolled to ({x}, {y}) — {edge}. Page: {_page_line(title, href)}",
            details=details,
        )
    if action == "logs":
        return _bridge_logs_result(tool_call_id, params, result, details, title, href)
    if action == "click":
        navigation = "" if result.get("navigated") else " (no navigation)"
        return _text(
            tool_call_id,
            "browser",
            f"Clicked {params.selector.strip()}{navigation}. Page: {_page_line(title, href)}",
            details=details,
        )
    if action == "type":
        got = str(result.get("value", "")).strip()
        if got != params.text.strip():
            return _error(
                tool_call_id,
                "browser",
                f"fill of {params.selector.strip()} did not take: the field holds "
                f"{got!r}, not {params.text!r}. Check the selector names an editable "
                "field ('snapshot' shows what is there).",
            )
        return _text(
            tool_call_id,
            "browser",
            f"Typed into {params.selector.strip()}. Value is now {got!r}.",
            details={**details, "selector": params.selector.strip()},
        )

    # The extension returns bytes, never a filesystem path. The Python tool
    # keeps path resolution, PNG validation and write approval in one place.
    if params.path:
        resolved, _inside, _resolvable = _resolve_workspace_path(params.path, _safe_cwd(context))
        target = str(resolved)
    else:
        import tempfile

        safe_surface = re.sub(r"[^A-Za-z0-9_-]", "-", surface)
        target = os.path.join(tempfile.gettempdir(), f"lo-browser-{safe_surface}.png")
    try:
        payload = base64.b64decode(str(result.get("data", "")), validate=True)
    except ValueError:
        return _error(tool_call_id, "browser", "browser extension returned invalid screenshot data")
    if not payload.startswith(PNG_MAGIC):
        return _error(
            tool_call_id,
            "browser",
            f"browser extension capture is not a PNG ({len(payload)} bytes)",
        )
    try:
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_bytes(payload)
    except OSError as exc:
        return _error(tool_call_id, "browser", f"could not write screenshot to {target}: {exc}")
    return _text(
        tool_call_id,
        "browser",
        f"Screenshot of {_page_line(title, href)} saved to {target} ({len(payload)} bytes).",
        details={"path": target, "bytes": len(payload)},
    )


async def retitle_browser_surface(state: BrowserSurfaceProtocol, context: ToolContext) -> None:
    """Push a title that arrived after the tab was opened, so its group renames.

    Public because the SESSION calls it, not the model: a conversation names
    itself asynchronously, typically after the opening turn's ``open`` already
    created the group, and every other command only reconciles the group as a
    side effect of doing something else. A session that opens a tab, screenshots
    it and closes issues no such command, so without this its group wears the
    open-time label — the bare fallback — for the tab's whole life.

    Identity rides the same trusted ``_browser_identity_params`` boundary as
    every other command: the requester is derived from the host's context, never
    from an argument, so this cannot be used to rename another session's group.

    Entirely best-effort and never raised: grouping is presentation, and a
    rename must not cost the title, the turn, or a browse. Cmux surfaces have no
    group chrome to rename, so they are skipped rather than errored.
    """
    surface = state.surface_id
    if not surface.startswith("bridge:"):
        return
    # A rename REQUIRES a real session identity, so it declines rather than
    # borrowing ``_browser_requester``'s tool-call fallback. That fallback mints
    # ``call:<tool_call_id>``, which is unique only because a tool call id is;
    # this is not a tool call, so it would put the CONSTANT ``call:retitle`` in
    # an identity slot whose whole purpose is to be distinct per session (see
    # ``_browser_requester``). Harmless today — ``trustedOwner`` ignores
    # anything not prefixed ``session:``, so the reconcile would no-op — but a
    # shared constant sitting in an identity field is a trap for whoever
    # relaxes that check next (review round 1, R3). Nothing is lost by
    # declining: without a session id the extension could not attribute the
    # group anyway.
    if context is None or not context.session_id:
        return
    identity = _browser_identity_params(context, "retitle")
    _result, problem = await _bridge_call(
        "retitle", "retitle", {"tab": surface, **identity}, surface=surface
    )
    if problem is not None:
        # Logged, not surfaced: the tab is still perfectly usable under the
        # label it already has, and the next ordinary command reconciles it.
        logger.debug("browser: could not push the session title to the tab group")


async def close_browser_surface(state: BrowserSurfaceProtocol) -> str:
    """Close the recorded surface and drop the handle. Returns "" on success
    (or when there was nothing open), else cmux's diagnostic.

    Public because SESSION TEARDOWN calls it as well as the tool: the surface
    outlives the per-turn ToolContext, so a session that ends without the model
    thinking to say 'close' would otherwise strand a browser tab in the user's
    pane forever.

    The handle is dropped whatever the exit code says. If the surface is
    already gone — the user closed the tab, or cmux restarted — the call fails,
    and keeping a dead handle would point every later action at whatever tab is
    active (see :func:`_stale_surface_error`) with no route back, because
    'open' itself reuses the recorded handle.
    """
    surface = state.surface_id
    if not surface:
        return ""
    if surface.startswith("bridge:"):
        _result, problem = await _bridge_call(
            "teardown", "close", {"tab": surface}, surface=surface
        )
        state.surface_id = ""
        if problem is None:
            return ""
        return "browser extension could not close the tab"
    code, out = await _run_cmux(["close-surface", "--surface", surface])
    state.surface_id = ""
    return "" if code == 0 else out or f"cmux exited {code}"


async def _browser_close(tool_call_id: str, state: BrowserSurfaceProtocol) -> ToolResult:
    if not state.surface_id:
        return _text(tool_call_id, "browser", "No browser surface open.", useless=True)
    surface = state.surface_id
    problem = await close_browser_surface(state)
    if problem:
        return _text(
            tool_call_id,
            "browser",
            f"Browser surface {surface} could not be closed ({problem}); dropped the handle.",
        )
    return _text(tool_call_id, "browser", f"Closed browser surface {surface}.")


@_guard("browser")
async def execute_browser(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Drive cmux when present, else the paired Local Operator extension."""
    try:
        params = BrowserParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "browser", exc)
    action = params.action.strip().lower()
    if action not in BROWSER_ACTIONS:
        return _error(
            tool_call_id,
            "browser",
            f"unknown action: {action} (expected one of {', '.join(BROWSER_ACTIONS)})",
        )
    cmux_available = cmux_browser_available()
    # Classify the daemon ONCE per action and reuse that answer for both the
    # backend decision and the demotion diagnostic, so they can never describe
    # different readings of a file that may change between two reads.
    bridge_liveness = _bridge_liveness()
    # The socket-confirming probe, not the bare file check: a healthy daemon
    # whose heartbeat writer stopped must not silently demote this session to
    # cmux. It costs a round-trip only in the stale-but-alive case.
    bridge_available = await bridge_browser_reachable(classified=bridge_liveness)
    if not cmux_available and not bridge_available:
        return _error(
            tool_call_id,
            "browser",
            "browser not available: neither cmux nor a connected Local Operator browser extension "
            "is reachable. Run 'lop browser status' and 'lop browser install' to set "
            "up the bridge. Do not install or script one instead; a separate browser engine "
            "cannot preserve the user's real logins.",
        )
    # Before the state lookup and before every subprocess, including the
    # liveness probe below: see _validate_browser_args.
    problem = _validate_browser_args(action, params)
    if problem:
        return _error(tool_call_id, "browser", problem)
    state = _browser_state(context)

    # The access actions are dispatched BEFORE any surface logic: they exist
    # for the moment 'open' just failed with origin_not_allowed, so there is
    # usually no surface to key on, and gating them behind "no browser surface
    # open — use 'open' first" would send the agent in a circle. They need the
    # bridge (cmux has no permission model), so a cmux-pinned surface or a
    # bridge-less host degrades with the same typed error as scroll/logs.
    if action in ("request_access", "await_access", "cancel_access"):
        if state.surface_id.startswith("surface:") or not bridge_available:
            return _error(
                tool_call_id,
                "browser",
                f"'{action}' is not supported on the cmux backend — cmux has no "
                "site-permission prompts; navigation works directly. This action only "
                "exists for the Local Operator browser extension ('lop browser status' / "
                "'lop browser install')." + _bridge_demotion_hint(bridge_liveness),
            )
        return await _bridge_access(tool_call_id, action, params, context)

    # Backend is selected only for an empty surface. A prefixed handle pins the
    # transport, so a browser opening or closing mid-session cannot silently
    # move the agent to a different surface.
    if action == "open":
        # Backend precedence on a FRESH open: prefer the paired Local Operator
        # browser extension over cmux when both are reachable. The extension
        # drives a real Chromium profile with the user's own logins and — by
        # construction (nav.ts creates its tab with ``active: false`` and never
        # activates it, raises a window, or calls captureVisibleTab) — never
        # steals focus, so a background agent can browse while the user works in
        # another window. cmux remains a first-class FALLBACK: it is used when no
        # extension is connected, and an already-open cmux surface stays on cmux.
        #
        # A prefixed handle still pins the transport for the life of the surface,
        # so this only decides where a brand-new surface lands; a browser opening
        # or closing mid-session can never silently move the agent between
        # backends.
        if state.surface_id.startswith("bridge:"):
            return await _bridge_open(tool_call_id, state, params.url, context)
        if state.surface_id.startswith("surface:"):
            return await _browser_open(tool_call_id, state, params.url)
        if bridge_available:
            return await _bridge_open(tool_call_id, state, params.url, context)
        return await _browser_open(tool_call_id, state, params.url)
    if action == "close":
        return await _browser_close(tool_call_id, state)
    if action == "tabs":
        # Discovery works without an owned surface (its point is finding out
        # what is open), but it is extension-only: cmux keeps no multi-surface
        # registry, so a cmux-pinned session degrades exactly like scroll/logs.
        if state.surface_id.startswith("surface:") or not bridge_available:
            return _error(
                tool_call_id,
                "browser",
                "'tabs' is not supported on the cmux backend — use the Local Operator "
                "browser extension (run 'lop browser status' / 'lop browser install' to "
                "set it up). cmux has no multi-tab surface registry, so this action only "
                "works through the extension bridge." + _bridge_demotion_hint(bridge_liveness),
            )
        return await _bridge_tabs(tool_call_id, state)
    if not state.surface_id:
        return _error(tool_call_id, "browser", "no browser surface open — use 'open' first")
    if state.surface_id.startswith("bridge:"):
        return await _bridge_action(tool_call_id, state, action, params, context)
    # A cmux-backed surface cannot serve the extension-only actions. Degrade with
    # a clear, actionable error naming the extension rather than faking a partial
    # scroll or an empty log list — the operator asked for these to work on the
    # bridge and to fail honestly on cmux.
    if action in BRIDGE_ONLY_BROWSER_ACTIONS:
        return _error(
            tool_call_id,
            "browser",
            f"'{action}' is not supported on the cmux backend — use the Local Operator "
            "browser extension (run 'lop browser status' / 'lop browser install' to set it "
            "up). cmux has no console-log tap or background-tab scroll primitive, so this "
            "action only works through the extension bridge."
            + _bridge_demotion_hint(bridge_liveness),
        )
    # ONE liveness probe here rather than one per action body, and never inside
    # a poll loop: cmux answers a dead handle by silently retargeting the
    # ACTIVE surface with exit 0, so without this check 'read', 'snapshot',
    # 'click', 'type', 'screenshot' and 'goto' would each drive and report on
    # whatever tab the user happens to be looking at. See
    # :func:`_stale_surface_error` for the measurements.
    stale = await _stale_surface_error(tool_call_id, state)
    if stale is not None:
        return stale
    surface = state.surface_id

    if action == "goto":
        return await _browser_goto(tool_call_id, surface, params.url)
    if action == "read":
        return await _browser_read(tool_call_id, surface, params.selector)
    if action == "snapshot":
        argv = _surface_argv(surface, "snapshot", "--compact")
        if params.selector.strip():
            argv += ["--selector", params.selector.strip()]
        code, out = await _run_cmux(argv)
        if code != 0:
            return _error(tool_call_id, "browser", f"cmux snapshot failed: {out}")
        return _text(
            tool_call_id,
            "browser",
            truncate_output(out, BROWSER_TEXT_LIMIT_CHARS) or "(empty snapshot)",
            details={"surface_id": surface},
        )
    if action == "click":
        return await _browser_click(tool_call_id, surface, params.selector)
    if action == "type":
        return await _browser_type(tool_call_id, surface, params.selector, params.text)
    return await _browser_screenshot(tool_call_id, surface, params.path, context)


#: A cmux surface handle is exactly ``surface:<n>`` — the PREFIX is part of the
#: contract, not just the colon shape. Anchoring on it is what stops three
#: distinct classes of bad handle: a status banner ("done", "error: could not
#: start"), a sibling ref of the wrong kind (``pane:2``, ``window:1`` — both
#: real cmux output that the old shape-only regex happily adopted), and an
#: option-looking value like ``--help`` which would be injected straight into
#: the next ``cmux browser --surface <x> goto`` argv.
_SURFACE_REF_RE = re.compile(r"^surface:[0-9A-Za-z_-]+$")

#: JSON keys that may carry the handle, best first. ``id`` is deliberately NOT
#: here: it is a documented cmux field with an unrelated request-dedupe
#: meaning, so reading it turned an error payload like
#: ``{"ok":false,"error":"browser disabled","id":"req-8f21"}`` into a
#: confidently-reported success on a handle that does not exist.
_SURFACE_KEYS = ("surface_ref", "surface", "surface_id")


#: Bounds on parsing untrusted subprocess output. A real cmux handshake is a
#: few hundred bytes; these exist so a pathological payload costs a bounded
#: amount of work and degrades to "no handle" (an outcome the caller already
#: handles) rather than to seconds of CPU or an unexpected exception.
_MAX_PARSE_CHARS = 64 * 1024
_MAX_JSON_NODES = 10_000
_MAX_DECODE_ATTEMPTS = 256


def _iter_json_objects(out: str):
    """Yield every JSON object embedded anywhere in ``out``, in order.

    cmux's real ``--json`` output is a PRETTY-PRINTED multi-line document, but
    it can also arrive as NDJSON, wrapped in a human preamble, or followed by a
    trailing status line. Splitting on lines handles NDJSON and breaks
    pretty-printing; requiring the whole stream to be one document does the
    reverse. ``raw_decode`` from each ``{`` handles all four, because it stops
    at the end of the first complete value and tells us where that was.
    """
    decoder = json.JSONDecoder()
    index = 0
    attempts = 0
    while attempts < _MAX_DECODE_ATTEMPTS:
        start = out.find("{", index)
        if start < 0:
            return
        attempts += 1
        try:
            parsed, end = decoder.raw_decode(out, start)
        except (ValueError, RecursionError):
            # RecursionError as well as ValueError: the C JSON decoder recurses
            # per nesting level and raises it (NOT a ValueError subclass) on a
            # deeply nested payload. Untrusted subprocess output must degrade to
            # "no handle", which the caller already handles, never to an
            # unexpected-exception tool failure.
            # Not the start of a complete object — step past this brace rather
            # than giving up, so a literal "{" in a preamble cannot hide the
            # real payload behind it. Each failed attempt can rescan forward,
            # so the attempt count is what keeps a run of bare "{" from going
            # quadratic (60k of them cost seconds before this bound).
            index = start + 1
            continue
        if isinstance(parsed, dict):
            yield parsed
        index = max(end, start + 1)


def _find_surface_ref(payload: object) -> str:
    """Breadth-first search for a ref-shaped handle in a decoded JSON payload.

    Nested is normal (``{"ok":true,"result":{"surface_ref":"surface:73"}}``),
    so a flat key lookup missed real successes. Every candidate is still
    validated against :data:`_SURFACE_REF_RE`, so searching widens what we
    ACCEPT without widening what we TRUST.

    Iterative with an explicit queue rather than recursive: a recursive walk
    raised RecursionError at ~2000 levels of nesting, and tool output is
    untrusted input — a pathological payload must degrade to "no handle", which
    is already a handled outcome, not to an unexpected-exception tool failure.
    Breadth-first also finds the SHALLOWEST match, which is the one a sane
    payload means.
    """
    queue: list[object] = [payload]
    seen = 0
    while queue and seen < _MAX_JSON_NODES:
        current = queue.pop(0)
        seen += 1
        if isinstance(current, dict):
            for key in _SURFACE_KEYS:
                value = current.get(key)
                if isinstance(value, str) and _SURFACE_REF_RE.match(value.strip()):
                    return value.strip()
            queue.extend(current.values())
        elif isinstance(current, list):
            queue.extend(current)
    return ""


def _parse_surface_id(out: str) -> str:
    """Extract the surface handle from ``cmux --json new-surface`` output.

    Returns "" when nothing ``surface:<n>``-shaped is found, so callers fail
    with an honest error instead of poisoning the session with a handle that
    every later ``--surface`` call will silently misuse.
    """
    # Only the head of the output is searched. A real handshake is a few hundred
    # bytes; scanning megabytes of unrelated output for a brace is wasted work,
    # and `{` * 60000 made the decode-attempt loop take seconds.
    head = out[:_MAX_PARSE_CHARS]
    for payload in _iter_json_objects(head):
        found = _find_surface_ref(payload)
        if found:
            return found
    # Fallbacks, in order of how much structure they assume.
    #
    # (a) A ref-shaped bare token — plain-text output, or JSON whose braces the
    #     scan bound above could not reach.
    for token in head.split():
        if _SURFACE_REF_RE.match(token):
            return token
    # (b) A QUOTED ref under one of the known keys. This is what saves a
    #     legitimate payload larger than the scan window: truncating the head
    #     mid-document makes raw_decode fail, and (a) cannot match because the
    #     token still carries its JSON quotes and comma. Real cmux sends 118
    #     bytes so this is not reachable today, but a future response embedding
    #     a snapshot or a data-URI would otherwise lose EVERY handle at a sharp
    #     64 KB cliff — a total failure, not a degraded one.
    for key in _SURFACE_KEYS:
        match = re.search(rf'"{key}"\s*:\s*"(surface:[0-9A-Za-z_-]+)"', head)
        if match:
            return match.group(1)
    return ""


def build_browser_tool(context: ToolContext | None) -> AgentTool | None:
    """Advertise the browser tool when cmux or the extension bridge is reachable.

    Mirrors the wake builder: an environment-specific capability that returns
    None (excluded from the inventory) when the host cannot support it.

    There is deliberately no headless fallback. This repo ships no browser
    engine — playwright belongs to the pre-rewrite codebase and appears in no
    dependency group — and pulling one into the default install would add a
    ~150 MB browser download to a dependency set that is kept small on
    purpose. A host without cmux therefore has no browser tool at all, which
    is honest, and the agent still reaches static pages through `bash` and
    curl.

    The DESCRIPTION says what the surface is, not just which verbs it takes,
    because a verb list gave the model no reason to prefer it. Measured: a
    session that needed before/after screenshots of a local dev server wrote a
    playwright script and spent 23 s on `playwright install chromium` while
    this tool sat in its inventory, and then — told outright to "use the cmux
    browser instead" — still shelled the cmux CLI through `bash` rather than
    calling it. The deciding fact is persistence: this drives the browser the
    user is looking at, so its cookies and logins survive between calls and
    between sessions and the user can sign in by hand when asked, which is
    exactly what a freshly downloaded headless Chromium can never do.

    Backend precedence (see ``execute_browser``): when both a cmux panel and a
    paired Local Operator browser extension are reachable, a fresh open prefers
    the EXTENSION. It drives a real Chromium profile with the user's own logins
    and never steals focus — its tab is created inactive and no action raises a
    window — so a background agent can browse while the user works elsewhere.
    cmux and (outside this tool) playwright are fallbacks for hosts without the
    extension. The full setup/permissions playbook lives in ``guide://browser``.
    """
    # Gating deliberately uses the WEAKER `advertisable` test, not the
    # backend-selection one: a stale-but-alive daemon must still put the tool
    # in the inventory so `execute_browser`'s bounded socket probe can acquit
    # it (or produce the typed demotion diagnostic). With the strict check
    # here, an extension-only host whose heartbeat writer had died offered no
    # browser tool at all for the whole session — no fallback and no way to
    # discover the healthy daemon. Still file-only and still synchronous: this
    # runs while constructing every session and opens no socket.
    if not cmux_browser_available() and not bridge_browser_advertisable():
        return None
    return AgentTool(
        name="browser",
        label="Browser",
        describe_approval=_describe_browser_approval,
        description=(
            "Drive the user's REAL browser (a cmux browser panel or their paired "
            "Local Operator browser extension): open/goto a URL, read page text, snapshot the "
            "accessibility tree for click refs, click, type, scroll, read console "
            "logs, screenshot, close. Cookies and logins persist across calls and "
            "across sessions, and the user can sign in by hand when you ask them "
            "to, so this reaches authenticated pages a throwaway headless browser "
            "cannot. 'scroll' pages the view (default one screen down, or by "
            "x/y pixels, a direction keyword, or a selector to reveal) and reports "
            "whether more content remains; 'logs' returns the page's console "
            "output and uncaught exceptions for debugging web apps. Parallel "
            "sessions each drive their own tab: a fresh 'open' creates one NEW "
            "tab owned by this session; reuse it because later opens navigate it. "
            "Before your final response, call 'close' unless the user explicitly "
            "needs it left open for a pending or immediately continuing interaction. "
            "'tabs' lists every extension-driven tab including other sessions' "
            "(handles are redacted: the listing is awareness-only and cannot "
            "drive or close anything), and 'close' ends only your own tab. "
            "'scroll', 'logs' and "
            "'tabs' need the extension backend (cmux says so). On the extension, "
            "'open'/'goto' to a site the user has not approved fails with "
            "origin_not_allowed: then call 'request_access' with the url, NOTIFY the "
            "user (ask tool or message) to approve the prompt in the extension popup, "
            "and 'await_access' to wait for their decision before navigating again. "
            "Use it for every "
            "screenshot and page interaction; never install or script a browser "
            "engine instead."
        ),
        parameters=BrowserParams.model_json_schema(),
        # Navigates and can write a screenshot file, so it rides the write
        # approval gate rather than auto-approved read.
        approval_tier="write",
        concurrency="shared",
        interruptible=False,
        execute=execute_browser,
    )


# ---------------------------------------------------------------------------
# task / wait / jobs — background subagent engine tools
# ---------------------------------------------------------------------------
# These three tools are the model-facing surface of the background job
# engine. They are createIf-gated exactly like ``wake``: the ``task`` builder
# returns None unless the ToolContext carries a ``subagent_launcher`` (the
# session's ``Session._launch_subagent``), and ``wait``/``jobs`` return None
# unless the context carries a job manager. A session without the engine must
# never advertise tools that can only error.


class TaskItem(BaseModel):
    """One slice of a task batch. ``agent`` names the ROLE the child runs as —
    a registered profile or a packaged starter (reviewer, coder, architect,
    manager, designer, scout); the role supplies standing guidance and may
    restrict the child's tools. ``effort`` routes to a configured model tier
    (values.subagents.models lo/med/hi)."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(description="Short label for this subagent (shown in the jobs list).")
    prompt: str = Field(description="The full instructions this subagent runs.")
    # Free-form rather than a Literal: roles are user data, so the valid set is
    # whatever the operator's registry holds plus the packaged starters, and
    # baking today's names into the schema would make an operator-authored role
    # unlaunchable. An unknown name degrades to a full child (see
    # ``harness.subagent._resolve_role``) rather than failing the launch.
    agent: str = Field(
        default="task",
        description=(
            "Role for this subagent: 'task' (full child, no role), 'scout' "
            "(read-only research), or any role from the `agent` tool — e.g. "
            "'reviewer', 'coder', 'architect', 'manager', 'designer'. A role "
            "carries vetted guidance and may restrict tools."
        ),
    )
    effort: Literal["lo", "med", "hi"] | None = Field(
        default=None,
        description="Model tier for this subagent (values.subagents.models).",
    )


class TaskParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str | None = Field(
        default=None,
        description="Single-task form: short label for the subagent.",
    )
    prompt: str | None = Field(
        default=None,
        description="Single-task form: the instructions the subagent runs.",
    )
    # Mirrors of the TaskItem fields for the SINGLE-task form. Without these,
    # ``task(label=..., prompt=..., agent="reviewer")`` — the most natural way
    # to ask for one reviewer — fails schema validation (extra="forbid"), and
    # the caller has to discover that a role is only reachable through the
    # batch form. Observed live: a model hit exactly that error, then retried
    # with ``tasks=[...]``, paying a wasted round trip to learn it.
    agent: str | None = Field(
        default=None,
        description="Single-task form: role for the subagent (see 'tasks[].agent').",
    )
    effort: Literal["lo", "med", "hi"] | None = Field(
        default=None,
        description="Single-task form: model tier for the subagent.",
    )
    context: str = Field(
        default="",
        description=(
            "Shared context prepended to EVERY task in the batch — the goal, "
            "constraints, and interfaces every subagent needs. Stated once "
            "here instead of copy-pasted into each prompt."
        ),
    )
    tasks: list[TaskItem] = Field(
        default_factory=list,
        description=(
            "Batch form: all items launch as CONCURRENT subagents from this "
            "one call. Independent slices belong here together — one round "
            "trip instead of one per task."
        ),
    )

    @model_validator(mode="after")
    def _one_form(self) -> "TaskParams":
        """Single form xor batch form, whole — the pydantic voice keeps the
        'invalid arguments' contract for a half-supplied call."""
        if (self.label is None) != (self.prompt is None):
            raise ValueError("single-task form needs both label and prompt")
        single = self.label is not None
        if single and self.tasks:
            raise ValueError("pass either 'tasks' (batch) or label/prompt, not both")
        if not single and (self.agent or self.effort):
            # Silently ignoring these would be worse than refusing: the caller
            # believes it asked for a role, and every child in the batch would
            # run without one.
            raise ValueError("in the batch form, set 'agent'/'effort' on each tasks[] item")
        if not single and not self.tasks:
            raise ValueError("nothing to launch: pass 'tasks' or label/prompt")
        if not single and not self.context.strip() and len(self.tasks) == 1:
            # Allowed, but the batch-of-one without context is just the
            # single form with extra steps; no error, the model learns from
            # the schema descriptions.
            pass
        return self


def _coerce_job_targets(value: Any) -> Any:
    """Accept the string shapes models emit for job ids instead of a bare id or array.

    Observed live in transcripts (e.g. with Gemini 3.8 Flash, or when quoting):
    models pass ``job_id`` as a stringified list ``'["id1", "id2"]'`` or
    unquoted bracketed string ``'[id1, id2]'``, or a single element list / string.
    Unwraps JSON lists, unquoted bracketed strings, and lists with stringified items.
    """

    def _parse_str(text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except ValueError:
                parsed = None
            if isinstance(parsed, list):
                res: list[str] = []
                for item in parsed:
                    if isinstance(item, str):
                        res.extend(_parse_str(item))
                return res
            inner = text[1:]
            if inner.endswith("]"):
                inner = inner[:-1]
            items = [item.strip().strip("'\"") for item in inner.split(",")]
            res = []
            for item in items:
                if item:
                    res.extend(_parse_str(item) if item.startswith("[") else [item])
            return res
        return [text]

    if isinstance(value, str):
        items = _parse_str(value)
        if len(items) == 1:
            return items[0]
        if len(items) > 1:
            return items
        return value
    if isinstance(value, (list, tuple)):
        items = []
        for x in value:
            if isinstance(x, str):
                items.extend(_parse_str(x))
            else:
                items.append(x)
        if len(items) == 1 and isinstance(items[0], str):
            return items[0]
        return items
    return value


def _coerce_single_job_id(value: Any) -> Any:
    """Unwrap a single job ID from string, bracketed string, or list."""
    if value is None:
        return None
    coerced = _coerce_job_targets(value)
    if isinstance(coerced, list):
        if len(coerced) == 1 and isinstance(coerced[0], str):
            return coerced[0]
        if len(coerced) == 0:
            return ""
        # If multiple were somehow provided to a single-id field, pick the first
        return coerced[0] if isinstance(coerced[0], str) else str(coerced[0])
    return coerced


def _resolve_job_target(target: str, jobs: Any, comms: Any = None) -> tuple[str | None, str | None]:
    """Resolve a target (canonical ID or label) to a canonical job ID.

    Priority:
    1. Direct match in jobs manager (ID).
    2. Subagent comms resolution if comms available (resolves label or ID).
    3. Label match among jobs in `jobs.list()`. If multiple match, prioritize running
       jobs (`job.status == 'running'`), matching comms.resolve behavior.
    Returns (job_id, error_message).
    """
    target = target.strip()
    if not target:
        return None, "empty target"
    if jobs.get(target) is not None:
        return target, None
    if comms is not None and hasattr(comms, "resolve"):
        try:
            resolved, error = comms.resolve(target)
            if error is None and len(resolved) == 1:
                return resolved[0], None
        except Exception:
            pass
    try:
        all_jobs = jobs.list()
    except Exception:
        all_jobs = []
    matches = [j for j in all_jobs if getattr(j, "label", None) == target]
    if len(matches) == 1:
        return matches[0].id, None
    if len(matches) > 1:
        live = [j for j in matches if getattr(j, "status", None) == "running"]
        if len(live) == 1:
            return live[0].id, None
        if len(live) > 1:
            return live[0].id, None
        return matches[0].id, None
    return None, f"unknown job {target}"


class WaitParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str | list[str] = Field(
        description=(
            "Job id from 'task' (or 'jobs'). Pass a LIST to wake on the first "
            "of several to finish — the way to await a fan-out without polling "
            "each child in turn."
        )
    )

    @field_validator("job_id", mode="before")
    @classmethod
    def _coerce_job_id(cls, value: Any) -> Any:
        return _coerce_job_targets(value)

    wait_ms: int = Field(
        # The ceiling and the default are sized to the work agents actually
        # await, not to a round-trip latency. Measured on this host's own
        # transcripts (30 h, Anthropic only): the old 300 000 ms cap made an
        # agent awaiting a CI pipeline (8-20 min) or a long subagent (30-90
        # min) poll every five minutes - 1 488 wait-only model calls in 877
        # consecutive chains, 434 of them two or more polls deep (up to 12).
        # Every poll re-sends the full context, and any poll landing after the
        # provider's 5-minute prompt-cache TTL rewrites the whole prefix: one
        # wait per chain would have saved 762 round trips (~207 M context
        # tokens) and 69 cache rewrites (20.8 M cache-write tokens, 8.7% of
        # all cache writes). A long budget strands nothing because the wait
        # is already interruptible: it returns on job settle, peer message
        # and steer (see ``execute_wait``), and the abort signal cuts it.
        # 60 min covers the longest subagent runs observed; 10 min is the
        # default because it spans a typical CI run without a second call.
        default=600_000,
        gt=0,
        le=3_600_000,
        # The sizing rule lives ONCE, in the tool description below: the two
        # are shipped together in every schema, so repeating it here bought
        # nothing but ~80 tokens on every turn (review round 1, R1-2).
        description="Max ms to block, up to 3600000 (60 min). Size it to the work.",
    )


class JobsParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["list", "peek", "cancel"] = Field(
        default="list",
        description=(
            "list: all jobs with status and age. peek: NEW output from one "
            "running job since your last peek. cancel: stop a running job."
        ),
    )
    job_id: str | None = Field(
        default=None,
        description="Job to peek at or cancel (required for those ops).",
    )

    @field_validator("job_id", mode="before")
    @classmethod
    def _coerce_job_id(cls, value: Any) -> Any:
        return _coerce_single_job_id(value)

    since: int | None = Field(
        default=None,
        description=(
            "For peek: resume from this 'seq' value (returned by the previous "
            "peek) so only new output comes back. Omit to read the tail from "
            "the start; pass 0 explicitly for the same."
        ),
    )


#: Formatted status text shared by ``wait``'s settled return and its detail
#: payload.  A child report can dwarf an ordinary tool result, so it uses the
#: same bounded, lossless spill path as every other verbose tool.
def _job_summary(job: Any, context: ToolContext | None = None) -> tuple[str, dict[str, Any] | None]:
    """Return a context-bounded handoff while keeping the full report readable.

    A task job's header names the model the child ran on, as recorded by the
    harness (``job.model_label``, set from the child's own model-change event),
    so the parent can STATE which model produced a delegated result rather
    than assume it. This is the parent-visible half of the pinned-tier work:
    ``subagent_start.model`` tells a stream consumer, this tells the model
    that launched the child. It is deliberately the harness's record and not
    anything the child said about itself, which is what makes it evidence
    when the question is whether a review was independent.
    """
    text = f"job {job.id} ({job.label}) [{job.status}]"
    model_label = str(getattr(job, "model_label", "") or "").strip()
    if getattr(job, "type", None) == "task" and model_label:
        text += f" model={model_label}"
    if job.status == "completed" and job.result_text:
        text += f"\n{job.result_text}"
    if job.status == "failed" and job.error_text:
        text += f"\n{job.error_text}"
    return spill_truncate(text, "wait", context)


@_guard("task")
async def execute_task(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Launch one subagent — or a whole batch — as background jobs.

    The batch form exists because fan-out economics are the point: N
    independent slices cost N sequential model round trips when launched one
    per call, and one call when launched as ``tasks`` together. The shared
    ``context`` is prepended to every prompt so the contract (goal,
    constraints, interfaces) is stated once by the delegator and cannot drift
    between children.
    """
    try:
        params = TaskParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "task", exc)

    launcher = context.subagent_launcher if context else None
    if launcher is None:
        return _error(
            tool_call_id,
            "task",
            "subagent launching is not available in this session (no engine attached).",
        )
    items: list[TaskItem] = list(params.tasks)
    if params.label is not None and params.prompt is not None:
        # ``agent`` falls back to TaskItem's own default rather than being
        # passed as None, which the field does not accept; ``effort`` is
        # already Optional there and passes through unchanged.
        items.append(
            TaskItem(
                label=params.label,
                prompt=params.prompt,
                agent=params.agent or "task",
                effort=params.effort,
            )
        )

    def _full_prompt(item: TaskItem) -> str:
        if not params.context.strip():
            return item.prompt
        return f"{params.context.strip()}\n\n---\n" f"Your task ({item.label}):\n{item.prompt}"

    launched: list[dict[str, Any]] = []
    failures: list[str] = []
    for item in items:
        try:
            job_id = launcher(item.label, _full_prompt(item), agent=item.agent, effort=item.effort)
        except Exception as exc:  # noqa: BLE001 — engine failure surfaces as an error result
            # An unavailable tier is the one launch failure the model is
            # tempted to "fix" by retrying at another tier, which is how a
            # pinned reviewer ends up running on the author's own model. Say
            # so in the result, so the correct next step (report the tier as
            # broken, or launch with NO effort and disclose that the child
            # inherits the parent's model) is in front of it rather than
            # something it has to infer from a bare exception string.
            tier = getattr(exc, "tier", None)
            if tier is not None:
                failures.append(
                    f"{item.label}: {exc}. Do NOT retry at a different effort tier — "
                    f"that runs the child on a different model than {tier!r} was "
                    f"chosen for. Either fix subagents.models.{tier} or launch "
                    f"without 'effort' and state that the child inherits this "
                    f"session's model."
                )
            else:
                failures.append(f"{item.label}: {exc}")
            continue
        launched.append({"job_id": job_id, "label": item.label, "agent": item.agent})
    if not launched:
        detail = failures[0] if failures else "no tasks to launch"
        return _error(tool_call_id, "task", f"could not launch subagent(s): {detail}")
    lines = [f"- {entry['label']} ({entry['agent']}): job {entry['job_id']}" for entry in launched]
    body = (
        f"launched {len(launched)} subagent(s) as concurrent background jobs:\n"
        + "\n".join(lines)
        + "\nuse 'wait' (job_id=...) to await one, or 'jobs' to list running work."
    )
    if failures:
        body += "\nfailed to launch: " + "; ".join(failures)
    return _text(
        tool_call_id,
        "task",
        body,
        details={"jobs": launched, "job_id": launched[0]["job_id"], "label": launched[0]["label"]},
    )


def build_task_tool(context: ToolContext) -> AgentTool | None:
    if context.subagent_launcher is None:
        return None
    return AgentTool(
        name="task",
        label="Subagent task",
        describe_approval=_describe_task_approval,
        description=(
            "Launch background subagents — one, or a whole concurrent batch "
            "('tasks' + shared 'context') in a single call. 'agent' names a "
            "role carrying vetted guidance (reviewer, coder, architect, "
            "manager, designer, scout — see the `agent` tool); effort picks a "
            "configured model tier."
        ),
        parameters=TaskParams.model_json_schema(),
        # Spawns autonomous child work, so it rides the write gate just like
        # scheduling a wake: the user approves starting the child.
        approval_tier="write",
        concurrency="exclusive",
        interruptible=False,
        execute=execute_task,
    )


#: What ``wait`` tells the model for each kind of inbound arrival that woke it,
#: keyed by the producer's ``CustomMessage.custom_type``. Spelled as literals
#: rather than imported: ``session.session`` and ``harness.comms`` both import
#: this module, so naming ``PEER_MESSAGE_MESSAGE_TYPE`` / ``WAKE_PROMPT_MESSAGE_TYPE``
#: / ``HUB_MESSAGE_TYPE`` here would be a cycle. ``test_wait_budget.py`` pins
#: the three keys against the real constants so a rename cannot silently
#: demote a kind to the generic fallback wording.
_ARRIVAL_NOTES: dict[str, str] = {
    "peer_message": "a message arrived from another session",
    "wake_prompt": "a scheduled wake fired — read the reminder before re-waiting",
    "hub_message": "a hub message arrived from a subagent or your parent",
}


@_guard("wait")
async def execute_wait(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Block until a background job settles or ``wait_ms`` elapses."""
    try:
        params = WaitParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "wait", exc)

    jobs = context.jobs if context else None
    if jobs is None:
        return _error(
            tool_call_id,
            "wait",
            "job tracking is not available in this session (no job manager attached).",
        )
    raw_ids = [params.job_id] if isinstance(params.job_id, str) else list(params.job_id)
    raw_ids = [target for target in dict.fromkeys(raw_ids) if target]
    if not raw_ids:
        return _error(tool_call_id, "wait", "no job id given")

    comms = context.subagent_comms if context else None
    job_ids: list[str] = []
    missing: list[str] = []
    for target in raw_ids:
        canonical_id, _err = _resolve_job_target(target, jobs, comms)
        if canonical_id is not None and jobs.get(canonical_id) is not None:
            if canonical_id not in job_ids:
                job_ids.append(canonical_id)
        else:
            missing.append(target)

    if missing:
        return _error(tool_call_id, "wait", f"unknown job {', '.join(missing)}")

    def _settled() -> Any:
        """The first of the awaited jobs that is no longer running, or None."""

        for job_id in job_ids:
            job = jobs.get(job_id)
            if job is not None and job.status != "running":
                return job
        return None

    def _still_running() -> list[str]:
        """The awaited ids that are still running, in the order given.

        The unsettled branches report THIS rather than ``job_ids[0]``: on the
        multi-id path the first id is simply the first the caller passed and
        may itself have settled, so pinning it in ``details`` tells a caller
        that reads the payload (rather than parsing the text) about the wrong
        job.
        """

        return [
            job_id
            for job_id in job_ids
            if (row := jobs.get(job_id)) is not None and row.status == "running"
        ]

    deadline = time.monotonic() + params.wait_ms / 1000.0
    # Snapshot the peer count and RE-ARM the event before parking.
    #
    # Two halves, both load-bearing. The COUNT is what the wake decision reads
    # (not `is_set()`), so a message that landed between two waits, or before
    # this wait began, is still seen exactly once. The CLEAR is what stops a
    # stale set event from making `asyncio.wait` return instantly on every
    # iteration, which would spin this `while` loop at full speed until the
    # deadline — the same event-loop burn `_await_any_settled` documents for
    # evicted job rows, and worse than the poll it replaced.
    #
    # The producer only ever sets and increments; re-arming is the consumer's
    # job. No `await` separates the snapshot from the clear, so the sequence is
    # atomic with respect to the loop, and the producer runs on that same loop
    # (see PeerArrivalProtocol), so a message cannot slip between them and be
    # lost: it either lands before the snapshot and is counted, or after the
    # clear and re-sets the event. Do NOT "tidy" this by moving the count and
    # the clear adjacent to each other or by reordering them — what matters is
    # only that nothing awaits in between.
    peer = context.peer_arrival if context is not None else None
    peer_event = peer.event() if peer is not None else None
    peer_seen = peer.count() if peer is not None else 0
    # The per-kind snapshot rides the same atomic window as the count: it is
    # only ever DIFFED against the live tally after a wake, so the model can
    # be told which producer woke it (peer message, scheduled wake, hub note)
    # rather than blaming every wake on a peer.
    peer_kinds_seen = dict(peer.arrivals()) if peer is not None else {}
    if peer_event is not None:
        peer_event.clear()

    def _peer_interrupt(reason: str, arrivals: dict[str, int] | None = None) -> ToolResult:
        """The still-running payload for a wait cut short by a message/steer.

        Deliberately the SAME shape the deadline branch returns, so the model
        gets the job id and status back and can simply re-issue the wait. The
        jobs are untouched: nothing is cancelled and, critically, nothing is
        consumed, so auto-delivery still hands over the result later.

        ``arrivals`` is the per-kind count of what landed during the park
        (``reason == "inbound"``). The kinds are the producers' message
        types, so the note can say "a scheduled wake fired" for the user's
        reminder and "a subagent/parent hub message" for a child speaking up,
        and the model knows what to read before re-issuing the wait.
        """

        running = _still_running()
        # The note must agree with details["interrupted_by"]: the text is what
        # the model actually reads, so rendering "steering" for a cancel we
        # cannot attribute would keep the mislabelled claim alive in the one
        # place it matters (review finding N1).
        details: dict[str, Any] = {
            "job_id": (running or job_ids)[0],
            "status": "running",
            "interrupted_by": reason,
        }
        if reason == "inbound":
            kinds = arrivals or {}
            notes = [_ARRIVAL_NOTES.get(kind, f"a {kind} message arrived") for kind in kinds]
            note = "; ".join(notes) or _ARRIVAL_NOTES["peer_message"]
            # One kind is the common case, and naming it keeps the existing
            # machine contract (``interrupted_by == "peer_message"``) intact
            # for peer-only wakes; several at once stay "inbound" and the
            # breakdown rides ``arrivals``.
            if len(kinds) == 1:
                details["interrupted_by"] = next(iter(kinds))
            details["arrivals"] = dict(kinds)
        else:
            note = "the wait was cancelled"
        return _text(
            tool_call_id,
            "wait",
            f"job {', '.join(running or job_ids)} still running ({note})",
            details=details,
        )

    job = _settled()
    while job is None:
        if signal is not None and signal.aborted:
            running = _still_running()
            return _text(
                tool_call_id,
                "wait",
                f"job {', '.join(running or job_ids)} still running (wait aborted)",
                details={"job_id": (running or job_ids)[0], "status": "running"},
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            running = _still_running()
            return _text(
                tool_call_id,
                "wait",
                f"job {', '.join(running or job_ids)} still running after {params.wait_ms}ms",
                details={"job_id": (running or job_ids)[0], "status": "running"},
            )
        try:
            await _await_any_settled(jobs, job_ids, remaining, signal, peer_event)
        except asyncio.CancelledError:
            # `lop send --now` steers, which makes interruptible_runner cancel
            # this tool task and hand the model SKIPPED_RESULT_TEXT - losing
            # the job id and telling it the wait never happened. Reporting the
            # still-running payload ourselves makes --now and the mailbox path
            # return the same shape.
            #
            # ABORT MUST STAY STRONGER THAN STEERING. Esc sets signal.aborted
            # and expects the tool to stop; a tool that swallows
            # CancelledError unconditionally defeats that contract outright
            # (see the abort/steer split in harness/loop.py). So re-raise when
            # the signal is aborted, and only absorb the plain steer cancel.
            if signal is not None and signal.aborted:
                raise
            # A cancel with NO signal at all cannot be a steer: steering rides
            # interruptible_runner, which always passes one. Absorbing it here
            # would invent a steering event on a host that has no steering, so
            # let it propagate as the plain cancellation it is.
            if signal is None:
                raise
            # Steering is not the only remaining cancel source - a batch
            # teardown (`GeneratorExit` in _execute_batch's finally) cancels
            # the task with a live, non-aborted signal too, and the tool
            # cannot tell the two apart from here: ToolContext deliberately
            # exposes no steering capability, and adding one just to label a
            # string would put loop state in a tool for no behavioural gain.
            # `interrupted_by` is machine-readable and "steering" is the one
            # value implying a human or peer acted, so report the neutral,
            # always-true cause instead of guessing. The still-running payload
            # (the point of this branch) is unchanged either way.
            return _peer_interrupt("cancelled")
        # Settle BEFORE peer: when a job settles on the same loop iteration as
        # a message arrives, the finished result is strictly more valuable than
        # "a message arrived", and reporting the peer branch would describe a
        # completed job as running (and, via _still_running() == [], pin the
        # WRONG job id in details - the exact failure _still_running's
        # docstring exists to prevent). The message is not lost either way: it
        # is already parked in the session journal and lands at the next
        # turn-safe boundary.
        job = _settled()
        if job is None and peer is not None and peer.count() > peer_seen:
            arrivals = {
                kind: total - peer_kinds_seen.get(kind, 0)
                for kind, total in peer.arrivals().items()
                if total > peer_kinds_seen.get(kind, 0)
            }
            return _peer_interrupt("inbound", arrivals)
        if job is None and all(jobs.get(job_id) is None for job_id in job_ids):
            return _error(tool_call_id, "wait", f"job {', '.join(job_ids)} disappeared")
    # Handing the result to the model HERE means auto-delivery must not
    # repeat it when the session next goes idle (see Session._on_job_completed
    # and AsyncJob.consumed).
    cast(Any, jobs).mark_consumed(job.id)
    text, spill_details = _job_summary(job, context)
    if len(job_ids) > 1:
        still = [
            job_id
            for job_id in job_ids
            if job_id != job.id
            and (other := jobs.get(job_id)) is not None
            and other.status == "running"
        ]
        if still:
            text += f"\n({len(still)} still running: {', '.join(still)})"
    details = {"job_id": job.id, "status": job.status}
    if spill_details:
        details.update(spill_details)
    return _text(
        tool_call_id,
        "wait",
        text,
        details=details,
    )


async def _await_any_settled(
    jobs: Any,
    job_ids: list[str],
    remaining: float,
    signal: AbortSignal | None = None,
    peer_event: asyncio.Event | None = None,
) -> None:
    """Sleep until one of ``job_ids`` settles, the wait is aborted, a peer
    message arrives, or ``remaining`` seconds pass.

    Event-driven where the host supports it. The measured problem this fixes:
    across recorded sessions, 70% of ``wait`` calls hit their deadline, and the
    old implementation spent those waits re-reading a status field every 50 ms
    — 6000 wakeups per five-minute wait, all on the one event loop shared by
    the parent turn, every sibling child, and the TUI repaint.

    The ABORT is raced alongside the settle events, not checked between
    sleeps. The old 50 ms poll re-read ``signal.aborted`` on every tick, so
    parking on the settle events alone silently made the abort branch dead for
    a job that never settles: an aborted wait sat for its whole budget (then
    five minutes, now up to an hour) instead of returning. The TUI masks that
    through its own interruptible-tool poll, but that is a different mechanism
    and does not cover deadline-tripped signals or non-TUI embedders relying
    on the documented ``AbortSignal`` contract.

    Falls back to the poll loop when the manager predates ``settled_event``
    (a third-party job manager satisfying the older protocol must keep
    working), and the poll there is 100 ms rather than 50 ms because nothing
    observes the difference: the caller is a model waiting on a job that runs
    for minutes.
    """

    getter = getattr(jobs, "settled_event", None)
    if getter is None:
        await asyncio.sleep(min(0.1, remaining))
        return
    try:
        # Only ids that STILL HAVE A ROW. An id whose row the retention sweep
        # evicted mid-wait gets a pre-set event (nothing will ever settle it),
        # which would return from `asyncio.wait` immediately and spin the
        # caller's `while` loop at full speed until its deadline — burning the
        # event loop this function exists to protect, and faster than the poll
        # it replaced. Skipping those ids parks on the siblings that can still
        # fire; when NONE can, there is nothing to wait for and the caller's
        # own disappeared/timeout branches are the right answer, so sleep out
        # the remainder rather than returning into a hot loop.
        events = [getter(job_id) for job_id in job_ids if jobs.get(job_id) is not None]
    except Exception:  # noqa: BLE001 - a manager that cannot make events polls
        await asyncio.sleep(min(0.1, remaining))
        return
    if not events and signal is None and peer_event is None:
        # The peer event has to be in this condition: with no live job rows
        # and no signal, sleeping out the remainder here would silently drop
        # the peer waiter and leave the mailbox wake dead on the no-jobs path.
        await asyncio.sleep(remaining)
        return
    waiters = [asyncio.ensure_future(event.wait()) for event in events]
    if signal is not None:
        waiters.append(asyncio.ensure_future(signal.wait()))
    if peer_event is not None:
        # Raced alongside the settle events for the same reason the abort is:
        # checking it between sleeps would make it observable only after some
        # OTHER wake source fired, which on a job that never settles is never.
        waiters.append(asyncio.ensure_future(peer_event.wait()))
    try:
        await asyncio.wait(waiters, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
    finally:
        # Every waiter is cancelled, including the one that completed: leaving
        # a pending future behind on the timeout path would leak one task per
        # wait call for the life of the session.
        for waiter in waiters:
            waiter.cancel()
        await asyncio.gather(*waiters, return_exceptions=True)


def build_wait_tool(context: ToolContext) -> AgentTool | None:
    if context.jobs is None:
        return None
    return AgentTool(
        name="wait",
        label="Wait for job",
        # This description is the ONE place the sizing rule is stated in full
        # (system.md points here; the agents guide carries the why). The
        # model reads it at decision time, which is where the rule has to be.
        description=(
            "Block until a background job settles (or wait_ms elapses), returning "
            "its final output/status. Pass a LIST of job ids to wake on the first "
            "one to finish. Returns the moment work settles, a message arrives "
            "(peer, scheduled wake, subagent), or you are steered, so SIZE THE "
            "BUDGET TO THE WORK: estimate how long the job should take (CI run, "
            "review, build) and wait for all of it in one call, up to 60 min; "
            "an expired wait means check on the job, not re-poll. Short budgets "
            "only to manage progress mid-run (a training loop), with `jobs "
            "op='peek'` between them."
        ),
        parameters=WaitParams.model_json_schema(),
        # read-only observation of job state; blocks the turn but changes nothing.
        approval_tier="read",
        concurrency="exclusive",
        interruptible=True,
        execute=execute_wait,
    )


def _peek_job(
    tool_call_id: str,
    jobs: Any,
    job: Any,
    since: int,
    context: ToolContext | None,
) -> ToolResult:
    """Return only what a job has printed since ``since``.

    The incremental contract is what makes polling affordable. A caller
    watching a 40-minute training run peeks every so often; returning the whole
    tail each time would re-send the same bytes on every poll, and because each
    result is appended to the transcript verbatim it would also grow the
    context by the same bytes repeatedly. Returning only the delta means a
    quiet job costs one short line, and a busy one costs exactly what it
    actually produced.

    The ``seq`` in the reply is the cursor for the NEXT peek, so the caller
    never has to reason about offsets — it echoes back what it was given.
    """
    reader = getattr(jobs, "read_output", None)
    if reader is None:
        return _error(
            tool_call_id,
            "jobs",
            "this session's job manager does not record live output.",
        )
    window = reader(job.id, since)
    if window is None:
        return _error(tool_call_id, "jobs", f"unknown job {job.id}")
    text, seq, gap = window

    status = job.status
    header = f"job {job.id} [{status}] seq={seq}"
    if status != "running":
        # A settled job's full result is already on its way to the caller (or
        # readable through `wait`), so peek does not duplicate it; it says the
        # watching is over, which is the fact that changes what to do next.
        header += " — finished; use 'wait' for its result"
    parts = [header]
    if gap:
        parts.append(
            "[warning: output between your cursor and this window was dropped "
            "from the buffer — this excerpt is not contiguous with your last peek]"
        )
    if text:
        parts.append(text)
    elif status == "running":
        parts.append("(no new output since last peek)")
    body = "\n".join(parts)
    # Same bounded/spill path as every other verbose tool: a job that dumped
    # more than the per-call budget between two peeks stays readable without
    # letting one peek blow the result budget.
    summary, spill_details = _capped_list_body(
        body, body[:TOOL_OUTPUT_LIMIT_CHARS], "jobs", context
    )
    details: dict[str, Any] = {
        "job_id": job.id,
        "status": status,
        "seq": seq,
        "new_chars": len(text),
        "gap": gap,
    }
    if spill_details:
        details.update(spill_details)
    return _text(tool_call_id, "jobs", summary, details=details)


@_guard("jobs")
async def execute_jobs(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """List background jobs, peek at a running one's output, or cancel one."""
    try:
        params = JobsParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "jobs", exc)

    jobs = context.jobs if context else None
    if jobs is None:
        return _error(
            tool_call_id,
            "jobs",
            "job tracking is not available in this session (no job manager attached).",
        )

    if params.op in ("peek", "cancel"):
        if not params.job_id:
            return _error(tool_call_id, "jobs", f"op='{params.op}' requires job_id")
        # Deliberately NOT owner-scoped, and the same choice ``op="list"``
        # already makes. Scoping these by ``context.job_id`` looked like
        # defence in depth and was a regression: ``run_subagent`` registers
        # every ``task`` job with ``owner_id=None``, so inside a child session
        # a scoped lookup misses its own grandchildren — the tool listed a job
        # and then called that same id "unknown job". The isolation it was
        # meant to add is structural rather than per-call anyway: each
        # ``Session`` builds its own ``AsyncJobManager`` and nothing reassigns
        # a child's, so a child's manager never holds its parent's rows and
        # there is no cross-session id to reach in the first place.
        comms = context.subagent_comms if context else None
        target = params.job_id
        canonical_id, _err = _resolve_job_target(target, jobs, comms)
        job = jobs.get(canonical_id) if canonical_id else None
        if job is None:
            return _error(tool_call_id, "jobs", f"unknown job {params.job_id}")
        effective_id = job.id
        if params.op == "cancel":
            cancelled = await jobs.cancel(effective_id)
            if not cancelled:
                # cancel() refuses a job that already settled, which is not a
                # failure worth erroring on: the caller wanted it stopped and
                # it is stopped. Report the terminal status so the caller does
                # not retry a cancel that can never succeed.
                return _text(
                    tool_call_id,
                    "jobs",
                    f"job {effective_id} was not cancelled (status: {job.status})",
                    details={"job_id": effective_id, "status": job.status, "cancelled": False},
                )
            return _text(
                tool_call_id,
                "jobs",
                f"cancelled job {effective_id} ({job.label})",
                details={"job_id": effective_id, "cancelled": True},
            )
        return _peek_job(tool_call_id, jobs, job, params.since or 0, context)

    rows = jobs.list()
    if not rows:
        return _text(tool_call_id, "jobs", "no background jobs", details={"count": 0})
    now = time.time()
    lines = []
    for job in rows:
        # A settled job is reported by when it SETTLED (the useful fact: "it
        # finished N seconds ago"); a running one by how long it has been going.
        #
        # The running case read ``now - now`` and printed 0.0s for EVERY live
        # job, whatever its real age — the one number this tool exists to
        # report, and the reading a caller uses to decide whether a subagent is
        # progressing or wedged. A six-minute child and one launched a second
        # ago were indistinguishable.
        #
        # Two quantities in one column, so each row says WHICH with a sense
        # word: ``up`` is "has been running this long", ``ago`` is "settled
        # this long ago". Without it the two readings are identical in shape
        # and a reader comparing a settled row against the subagent panel
        # (which reports the job's own duration) gets two numbers and no way
        # to reconcile them.
        #
        # Eight cells, not six: a running bash job in a long session reaches
        # 2h46m40s, at which point ``6.1f`` overflows its field and shears the
        # label column one cell for that row alone. Eight covers 11.5 days.
        # ``start_time`` is guarded because ``JobManagerProtocol.list()`` is
        # typed ``list[Any]``, so a third-party embedder may hand this loop a
        # duck-typed row without one. The guard covers ONLY that attribute:
        # ``id``/``status``/``settled_at``/``label`` are read unguarded on the
        # same row, so a row missing any of those still raises.
        # The SENSE follows the status, never the clock. Sharing one test let
        # a settled row with no ``settled_at`` print ``up`` beside a
        # ``completed`` or ``cancelled`` — a contradiction a reader cannot
        # reconcile, reachable through the real manager in the window inside
        # ``cancel()``'s await where the status is set and the settle stamp is
        # not yet. A settled row with no clock says ``old``: settled, and when
        # is not known.
        running = job.status == "running"
        reference = (
            job.settled_at if not running and job.settled_at else getattr(job, "start_time", None)
        )
        # A PARKED job is ``running`` with ``queued=True`` and a runner that
        # has never been entered, so ``up`` would present its wait as uptime —
        # the same misreport this PR was filed to stop, on the third surface.
        # ``waiting`` names what the number is: time spent at the gate. The
        # check is guarded like ``start_time`` because this row may be
        # duck-typed by an embedder.
        if running and getattr(job, "queued", False):
            sense = "wait"
        elif running:
            sense = "up"
        else:
            sense = "ago" if job.settled_at else "old"
        # A row with no clock says so. Printing 0.0s made it byte-identical to
        # a job launched this instant — the exact unreadable number this tool
        # was fixed to stop printing. Both branches are nine cells, so the
        # grid holds either way.
        age = f"{max(now - reference, 0.0):8.1f}s" if reference else f"{'unknown':>9}"
        lines.append(f"{job.id}  {job.status:<9}  {age} {sense:<4}  {job.label}")
    return _text(
        tool_call_id,
        "jobs",
        "\n".join(lines),
        details={"count": len(rows)},
    )


def build_jobs_tool(context: ToolContext) -> AgentTool | None:
    if context.jobs is None:
        return None
    return AgentTool(
        name="jobs",
        label="List jobs",
        description=(
            "Inspect background jobs (task/bash). op='list' shows every "
            "running and recently-settled job with its id, status and age — "
            "'up' is how long a running job has been going, 'wait' is how long "
            "a parked one has been waiting for a slot, 'ago' is how long since "
            "a settled one finished, and 'old' is a settled job whose finish "
            "time was not recorded. op='peek' (job_id, and since=<seq from the "
            "last peek>) returns ONLY the output produced since that cursor, so "
            "polling a long job stays cheap — use it to watch a build, a "
            "training run, a terraform apply, or a pipeline poll without "
            "blocking. op='cancel' (job_id) stops a running job and kills its "
            "process tree."
        ),
        parameters=JobsParams.model_json_schema(),
        approval_tier="read",
        concurrency="shared",
        interruptible=False,
        execute=execute_jobs,
    )


# ---------------------------------------------------------------------------
# hub — parent↔subagent messaging and control
# ---------------------------------------------------------------------------
# One tool, two shapes, chosen by who is being built for (see
# ``build_hub_tool``). ONE tool rather than eight (list/peek/send/ask/steer/
# pause/cancel/resume) because they share a target and differ only in intent —
# eight entries would spend eight tool-schema slots and eight descriptions on
# one concept, and the model would still have to learn which of them means
# "and wait for the answer". Named ``hub`` after the surface the same ops have
# in omp, whose shape this follows deliberately: ``to`` addresses one peer or
# ``"all"``, delivery returns per-recipient receipts, and asking is a send
# that waits. ``peek`` is the read-only member of the family: it observes a
# child's transcript instead of acting on the child, which is why it needs no
# message and never interrupts the child.
#
# The two shapes are not cosmetic. A parent may address, redirect, stop and
# resume its children; a child has exactly one peer (its parent) and no
# children of its own, so it gets a tool with no ``op`` and no ``to`` at all.
# Advertising the parent schema to a child would spend the child's context on
# four ops it cannot use and invite it to try them.


def _coerce_hub_to(value: Any) -> Any:
    """Accept the string shapes models emit for ``to`` instead of an array.

    Observed live (2026-08-19, session "Fix analytics widget spacing and
    alignment"): a parent model retried ``op='ask'`` five times against a
    running reviewer and never once emitted a real array — ``"<id>"`` (a bare
    id), ``'["<id>"]'`` (the array JSON-serialized into a string, the form the
    TUI then prints so it *looks* like a list), and even ``'[<id>]'`` without
    quotes. Each failed ``to: Input should be a valid list`` and the turn
    burned on retries while the child kept working unheard. The schema below
    must stay a plain array — an ``anyOf`` union of two real types is the one
    construct the provider matrix rejects — so the leniency lives here, in a
    before-validator: parse a JSON array, fall back to a bracket-stripped
    split, fall back to the bare string as a single target. Anything that is
    not one of those shapes passes through untouched and fails validation
    with the normal message.

    The bracket-stripped split — also taken when a leading-``[`` value is not
    valid JSON, e.g. ``'[job-1'`` — splits on commas, so a comma-bearing label
    sent in the unquoted form splits into fragments; the JSON form is the only
    shape that carries commas safely. That is an accepted leniency tradeoff on
    a path that previously hard-failed: the fragments fail resolution instead
    of the call failing validation, and the observed live shapes never carried
    commas. Non-string items inside a parsed JSON array are dropped, so
    ``'[null]'`` coerces to ``[]`` and fails with the normal "needs a 'to'
    target" message rather than a fabricated ``"None"`` target name.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except ValueError:
            # '[<id>]' with unquoted items is not JSON; the bracket-stripped
            # split below recovers it.
            parsed = None
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, str)]
        inner = text[1:]
        if inner.endswith("]"):
            inner = inner[:-1]
        items = [item.strip().strip("'\"") for item in inner.split(",")]
        return [item for item in items if item]
    return [text] if text else value


class HubParams(BaseModel):
    """Parent-side hub arguments."""

    model_config = ConfigDict(extra="forbid")

    op: Literal["list", "peek", "send", "ask", "steer", "pause", "cancel", "resume"] = Field(
        description=(
            "list: every subagent you launched with its status and whether it can be "
            "resumed — including finished, failed and paused ones the 'jobs' tool no "
            "longer shows. peek: READ the subagent's transcript (ranged, cheap) to see "
            "its current progress without spending its attention — the fast way to "
            "check on a running child. send: a note, no reply waited for. ask: a "
            "question, blocks for the subagent's answer. steer: change what it is doing "
            "(becomes part of its instructions). pause: stop it now but keep it "
            "resumable. cancel: stop it for good. resume: relaunch a stopped, paused or "
            "failed subagent against its own transcript so it continues where it left "
            "off; names several targets to fan one message out to a whole batch at once."
        )
    )
    # A plain array, NOT ``str | list[str]``: pydantic renders a union as
    # ``anyOf``, and this module's schemas reach Gemini verbatim as
    # ``function_declarations`` (providers/clients.py builds the body with
    # ``tool.parameters`` untouched). A construct one provider rejects would
    # fail every request in the session, not just the hub call — no builtin
    # here uses a non-nullable anyOf, and this is not the tool to be first.
    # Nullable rather than absent for op='list', which addresses nobody. The
    # anyOf this renders is the NULLABLE kind (``[array, null]``), the same
    # shape ``message`` below has always had and the one every provider in the
    # matrix accepts; the construct the comment above warns about is a
    # non-nullable union of two real types.
    to: list[str] | None = Field(
        default=None,
        description=(
            "Who to address: job ids from 'task'/'jobs'/'hub op=list', subagent "
            'labels, or ["all"] for every running subagent. Several ids address '
            "several subagents. 'ask' and 'peek' take exactly one; 'resume' fans one "
            "message out to every target you name, so a batch of failed subagents can "
            "be resumed in a single call. Omit for op='list', which addresses nobody."
        ),
    )

    # Models that serialize the array themselves send ``to`` as a string (a
    # bare id, or the JSON of the list); see ``_coerce_hub_to`` for the live
    # observation. Coercing here keeps the retry loop from burning the turn.
    @field_validator("to", mode="before")
    @classmethod
    def _coerce_to(cls, value: Any) -> Any:
        return _coerce_hub_to(value)

    message: str | None = Field(
        default=None,
        description=(
            "The body. Required for send/ask/steer, and for resume (what to do next); "
            "ignored by list/peek/pause/cancel."
        ),
    )
    timeout_ms: int = Field(
        # The default is a BUDGET for a busy child to finish its current step
        # and answer, not a round-trip latency: measured on real transcripts,
        # a child that answers takes p50 ~3 minutes from injection to reply
        # (it completes the tool batch it is in first). The old 120 s default
        # was below that median, so the modal outcome of op='ask' was a
        # timeout even when the child WAS answering. 300 s covers the p90 of
        # answering children while staying inside the 600 s schema maximum.
        default=300_000,
        gt=0,
        le=600_000,
        description="op='ask' only: how long to wait for the answer.",
    )
    range: str | None = Field(
        default=None,
        description=(
            "op='peek' only: which transcript steps to read, as 'start-end' or "
            "'start-' (1-based inclusive, stable across peeks). Omit for the last "
            "few steps; use steps= for 'the last N'."
        ),
    )
    steps: int | None = Field(
        default=None,
        ge=1,
        description=(
            "op='peek' only: shorthand for the last N steps (overrides range). "
            "Omit for the default (5)."
        ),
    )


class HubChildParams(BaseModel):
    """Child-side hub arguments: one peer, one direction, no ops."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(
        description="What to tell the parent agent. Answers its question when it asked one."
    )

    # Children see the PARENT's hub tool in their own transcripts (the parent
    # used it to launch and message them) and a large share of them mirror the
    # shape back — ``{"op": "send", "message": ...}`` — which ``extra="forbid"``
    # then rejects, so the reply never reaches the parent and the parent's
    # ``ask`` burns to a timeout. Measured on real transcripts: 7 rejected
    # answers against 0 accepted via the tool. The child surface genuinely has
    # no ops, so the parent-shaped keys are dropped rather than honoured; the
    # message still goes through, which is the whole intent.
    @model_validator(mode="before")
    @classmethod
    def _drop_parent_shaped_keys(cls, value: Any) -> Any:
        if isinstance(value, dict):
            parent_shaped = ("op", "to", "timeout_ms", "range", "steps")
            return {k: v for k, v in value.items() if k not in parent_shaped}
        return value


def _describe_hub_approval(args: dict[str, Any], cwd: str) -> str:
    """``<op> <target>: <body>`` — the act, who it hits, and what it says.

    All three matter to the decision and none of them is the parameter shape:
    stopping a subagent and asking it a question are different answers, and
    "all" versus one id is the difference between a note and a broadcast. The
    body is truncated because an approval row is read at a glance.
    """
    op = str(args.get("op") or "send")
    target = args.get("to")
    if isinstance(target, list):
        target = ", ".join(str(item) for item in target)
    target = " ".join(str(target or "").split())
    # Cell-bounded with the app's ellipsis (same reasoning as the send
    # describer): the character-count bound this used to carry rendered a CJK
    # body at more than twice its intended width and wrapped the prompt.
    body = _truncate_approval_body(" ".join(str(args.get("message") or "").split()))
    head = f"{op} {target}".strip()
    return f"{head}: {body}" if body else head


def _hub_targets(comms: Any, raw: Any) -> tuple[list[str], list[str]]:
    """Resolve the ``to`` argument to ``(job ids, errors)``, order preserved
    and duplicates dropped (``["all", "<id>"]`` must not message one child
    twice)."""
    requested = raw if isinstance(raw, list) else [raw]
    ids: list[str] = []
    errors: list[str] = []
    for item in requested:
        resolved, error = comms.resolve(str(item))
        if error is not None:
            errors.append(error)
        for job_id in resolved:
            if job_id not in ids:
                ids.append(job_id)
    return ids, errors


def _hub_list(tool_call_id: str, comms: Any) -> ToolResult:
    """Render the subagent roster for ``op='list'``.

    Every row states the one thing the caller acts on \u2014 whether the child can
    be resumed \u2014 rather than leaving it to be inferred from the status, since
    ``completed``, ``failed``, ``cancelled`` and ``paused`` are all resumable
    while ``running`` is not, which is the opposite of the intuitive reading.
    """
    rows = comms.roster()
    if not rows:
        return _text(
            tool_call_id,
            "hub",
            "no subagents launched in this session",
            # ``useless`` is mirrored into details as well as set on the
            # result: the flag drives the renderer, and compaction's pruning
            # pass reads the key — the same pairing the delivery path below
            # uses.
            details={"op": "list", "count": 0, "useless": True},
            useless=True,
        )
    lines = [f"{len(rows)} subagent(s):"]
    for row in rows:
        age = f", {row.age_s:.0f}s" if row.age_s is not None else ""
        extras = "resumable" if row.resumable else (row.detail or "not resumable")
        lines.append(f"- {row.label} ({row.job_id}): {row.status}{age} — {extras}")
        if row.resumable and row.detail:
            lines.append(f"    {row.detail}")
        # The session id only where it can be acted on. It is the id
        # ``--resume`` takes (NOT the job id on the line above), and this
        # roster is the only surface that shows it now that children are kept
        # out of the ``/resume`` picker. Printed for resumable rows alone: on a
        # row that cannot be resumed it is a string to mistake for the job id
        # rather than something to type.
        if row.resumable and row.session_id:
            lines.append(
                f"    transcript {row.session_id} (read it with lop --resume {row.session_id})"
            )
    if any(row.resumable for row in rows):
        lines.append("")
        lines.append(
            "Resume one with hub op='resume' and its JOB id, plus an instruction for "
            "what to do next \u2014 or name several JOB ids to resume a whole batch in "
            "one call. The transcript id above is not a job id: it opens the "
            "child's history for reading and starts no agent."
        )
    return _text(
        tool_call_id,
        "hub",
        "\n".join(lines),
        details={
            "op": "list",
            "count": len(rows),
            "children": [
                {
                    "job_id": row.job_id,
                    "label": row.label,
                    "status": row.status,
                    "resumable": row.resumable,
                    "session_id": row.session_id,
                }
                for row in rows
            ],
        },
    )


async def _hub_peek(tool_call_id: str, comms: Any, params: Any, ids: list[str]) -> ToolResult:
    """Render ``hub op='peek'``: a bounded slice of one child's transcript.

    The parent's observation path that costs the child nothing: unlike
    ``ask``, it neither waits on the child nor spends the child's attention,
    and unlike ``wait`` it does not block until the child settles. The window
    is deliberately small by default — the op exists so a parent can check
    progress without a transcript dump landing in its own context.
    """
    # ``ids`` is the resolution ``_execute_hub_parent`` already validated;
    # resolving twice would spend a second roster walk on the same answer.
    start: int | None = None
    end: int | None = None
    if params.range is not None:
        try:
            start, end = _parse_line_range(params.range)
        except ValueError as exc:
            return _error(tool_call_id, "hub", str(exc))
    window = await comms.peek(ids[0], start=start, end=end, steps=params.steps)
    if window.error is not None:
        return _error(tool_call_id, "hub", f"{window.label} ({window.job_id}): {window.error}")

    lines = [
        f"{window.label} ({window.job_id}) [{window.status}] — "
        f"{len(window.steps)} of {window.total} transcript step(s):"
    ]
    for step in window.steps:
        lines.append(f"{step.index:>4}  {step.heading}")
        if step.body:
            for body_line in step.body.splitlines():
                lines.append(f"      {body_line}")
    if window.steps:
        first, last = window.steps[0].index, window.steps[-1].index
        hints = []
        if first > 1:
            hints.append(f"range='1-{first - 1}' for earlier steps")
        if last < window.total:
            hints.append(f"range='{last + 1}-' to continue")
        if hints:
            lines.append("(" + "; ".join(hints) + ")")
    else:
        lines.append("(the transcript is empty so far)")
    return _text(
        tool_call_id,
        "hub",
        "\n".join(lines),
        details={
            "op": "peek",
            "job_id": window.job_id,
            "status": window.status,
            "total": window.total,
            "shown": [step.index for step in window.steps],
        },
    )


def _hub_receipt_lines(deliveries: list[Any]) -> list[str]:
    return [
        (
            f"- {delivery.label} ({delivery.job_id}): {delivery.outcome}"
            + (f" — {delivery.error}" if delivery.error else "")
        )
        for delivery in deliveries
    ]


@_guard("hub")
async def execute_hub(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Message, steer, stop or resume subagents (parent), or answer the parent
    (child)."""
    comms = context.subagent_comms if context else None
    if comms is None:
        return _error(
            tool_call_id,
            "hub",
            "agent messaging is not available in this session (no subagent engine).",
        )
    if comms.is_child(context.job_id if context else None):
        return await _execute_hub_child(tool_call_id, args, comms, context)
    return await _execute_hub_parent(tool_call_id, args, comms)


async def _execute_hub_child(
    tool_call_id: str,
    args: dict[str, Any],
    comms: Any,
    context: ToolContext | None,
) -> ToolResult:
    try:
        params = HubChildParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "hub", exc)
    job_id = context.job_id if context else None
    if job_id is None:  # unreachable: is_child() already required one
        return _error(tool_call_id, "hub", "this subagent has no job id to reply from.")
    outcome = comms.reply_to_parent(job_id, params.message)
    return _text(tool_call_id, "hub", outcome, details={"direction": "to_parent"})


async def _execute_hub_parent(
    tool_call_id: str,
    args: dict[str, Any],
    comms: Any,
) -> ToolResult:
    try:
        params = HubParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "hub", exc)

    if params.op == "list":
        return _hub_list(tool_call_id, comms)

    if params.op not in ("cancel", "pause", "peek") and not (params.message or "").strip():
        return _error(tool_call_id, "hub", f"op='{params.op}' needs a message.")

    if not params.to:
        return _error(
            tool_call_id,
            "hub",
            f"op='{params.op}' needs a 'to' target; use op='list' to see the subagents.",
        )

    ids, errors = _hub_targets(comms, params.to)
    if not ids:
        return _error(
            tool_call_id,
            "hub",
            "; ".join(errors) or "no subagent matched; use 'jobs' to list them.",
        )
    # A question and a peek each have exactly one subject — one reply to read,
    # one transcript window to render — so they refuse a fan-out rather than
    # silently acting on the first match. Resume is deliberately NOT here: each
    # resume spawns an INDEPENDENT new job replaying that target's own
    # transcript, so N targets produce N separate children with no shared reply
    # and no shared transcript to collide. Fanning one message out to a whole
    # batch of stopped/failed children (e.g. after a provider stall) is exactly
    # what resume should do, so it falls through to the loop below.
    if params.op in ("ask", "peek") and len(ids) > 1:
        return _error(
            tool_call_id,
            "hub",
            f"op='{params.op}' addresses one subagent at a time; got {len(ids)}.",
        )

    if params.op == "peek":
        return await _hub_peek(tool_call_id, comms, params, ids)

    message = params.message or ""
    if params.op == "ask":
        reply = await comms.ask(ids[0], message, params.timeout_ms)
        if reply.error is not None:
            return _error(tool_call_id, "hub", f"{reply.label} ({reply.job_id}): {reply.error}")
        if reply.timed_out:
            return _text(
                tool_call_id,
                "hub",
                f"{reply.label} ({reply.job_id}) did not answer within {params.timeout_ms}ms; "
                "it is still running and the question is in its context. Read its "
                "progress with hub op='peek' instead of asking again — peek costs "
                "the child nothing.",
                details={"op": "ask", "job_id": reply.job_id, "timed_out": True},
            )
        return _text(
            tool_call_id,
            "hub",
            f"{reply.label} ({reply.job_id}) replied:\n{reply.text}",
            details={"op": "ask", "job_id": reply.job_id, "reply": reply.text},
        )

    if params.op == "resume":
        # Fan-out: resume every target in turn. Unlike ask/peek, each resume
        # spawns an independent new job on its own transcript, so a batch of
        # stopped/failed children can be relaunched in one call. ``comms.resume``
        # returns a ``(new_job_id, error)`` tuple rather than a ``Delivery``, so
        # we collect per-target receipts here and format them like the
        # send/steer/cancel block below without borrowing the Delivery shape.
        resumed: list[tuple[str, str | None, str | None]] = [
            # (resumed-from id, new job id, error)
            (job_id, *comms.resume(job_id, message))
            for job_id in ids
        ]
        acted = [receipt for receipt in resumed if receipt[2] is None]
        header = (
            f"resume: {len(acted)}/{len(resumed)} subagent(s)"
            if resumed
            else "resume: nothing to do"
        )
        lines = [header]
        for from_id, new_job_id, error in resumed:
            label = comms.label_of(from_id)
            if error is None:
                # Each success carries the NEW job id it was resumed as; the
                # transcript-replay guidance is stated once in the footer
                # rather than repeated on every line.
                lines.append(f"- {label} ({from_id}): resumed as job {new_job_id}")
            else:
                lines.append(f"- {label} ({from_id}): failed \u2014 {error}")
        lines.extend(f"- {error}" for error in errors)
        if acted:
            # Kept from the single-target wording: a resumed child replays its
            # own transcript before it reads this instruction, so the parent
            # must 'wait' for it rather than assume it acted immediately.
            lines.append(
                "Each replays its own transcript before reading this instruction. "
                "Await them with 'wait'."
            )
        return _text(
            tool_call_id,
            "hub",
            "\n".join(lines),
            details={
                "op": "resume",
                # Unlike the send/steer/cancel block below, where "job_ids" is the
                # target list, a resume spawns fresh jobs: "job_ids" here holds the
                # NEW successfully-spawned ids, with their sources in "resumed_from".
                "job_ids": [new_id for _from, new_id, error in resumed if error is None],
                "resumed_from": [from_id for from_id, _new, error in resumed if error is None],
                "acted": len(acted),
                # Mirrored into details as well as the flag: every other useless
                # site in this module carries the key, and compaction's pruning
                # pass reads it from there.
                "useless": not acted,
            },
            # Nothing was resumed: a receipt list of pure failures is not an
            # observation the model should act on as if it had been heard.
            useless=not acted,
        )

    deliveries = []
    for job_id in ids:
        if params.op == "send":
            deliveries.append(comms.send(job_id, message))
        elif params.op == "steer":
            deliveries.append(comms.steer(job_id, message))
        elif params.op == "pause":
            deliveries.append(await comms.pause(job_id))
        else:
            deliveries.append(await comms.cancel(job_id))

    acted = [delivery for delivery in deliveries if delivery.outcome != "failed"]
    header = (
        f"{params.op}: {len(acted)}/{len(deliveries)} subagent(s)"
        if deliveries
        else f"{params.op}: nothing to do"
    )
    lines = [header, *_hub_receipt_lines(deliveries), *(f"- {error}" for error in errors)]
    return _text(
        tool_call_id,
        "hub",
        "\n".join(lines),
        details={
            "op": params.op,
            "job_ids": ids,
            "acted": len(acted),
            # Mirrored into details as well as the flag: every other useless
            # site in this module carries the key, and compaction's pruning
            # pass reads it from there.
            "useless": not acted,
        },
        # Nothing was reached: a receipt list of pure failures is not an
        # observation the model should act on as if it had been heard.
        useless=not acted,
    )


def build_hub_tool(context: ToolContext) -> AgentTool | None:
    if context.subagent_comms is None:
        return None
    if context.subagent_comms.is_child(context.job_id):
        return AgentTool(
            name="hub",
            label="Message parent",
            description=(
                "Send a message to the parent agent that delegated this task. Use it to "
                "answer a question it asked you, or to report unprompted that you are "
                "blocked, that the task is wrong, or that you found something it needs "
                "to know now rather than at the end."
            ),
            parameters=HubChildParams.model_json_schema(),
            # A child talking to its own parent starts nothing and touches
            # nothing; gating it would also mean a background child stalling
            # on an approval prompt nobody is watching.
            approval_tier="read",
            concurrency="shared",
            interruptible=False,
            execute=execute_hub,
        )
    return AgentTool(
        name="hub",
        label="Subagent hub",
        describe_approval=_describe_hub_approval,
        description=(
            "Talk to and control the subagents you launched with 'task': list them all "
            "with their status (including finished, failed and paused ones 'jobs' no "
            "longer shows), send a note, ask one a question and get its answer (use "
            "this to find out whether a quiet child is stuck), steer one onto a "
            "different course, pause one so it can be picked up later, cancel one, or "
            "resume a stopped, paused or failed one (or a whole batch of them at once) "
            "against its own transcript so it continues where it left off. Address them "
            'by job id, by label, or "all".'
        ),
        parameters=HubParams.model_json_schema(),
        # Write, like 'task' and 'wake': these ops redirect, kill and restart
        # autonomous work. The gate is per TOOL, not per op, so the tier is
        # the highest any op needs — 'resume' starts a child session, which is
        # exactly what 'task' asks the user to approve. The read-only ops
        # (list, peek) downgrade per call so observing children never prompts.
        approval_tier="write",
        call_approval_tier=lambda args: (
            "read" if str(args.get("op") or "") in ("list", "peek") else "write"
        ),
        # 'ask' blocks the turn on another agent's answer; running it beside
        # other tools would hold a shared slot for the whole timeout.
        concurrency="exclusive",
        interruptible=True,
        execute=execute_hub,
    )


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------
#
# Why this tool exists: without it a model that needs a decision writes the
# options as PROSE — observed verbatim as "(A) Drop email … (B) Escalate it
# properly … (C) You have context I don't" — and the user then has to retype an
# answer the agent has to re-parse. The transcript is a stream, not a form, so
# lettered prose is the only shape a model has; this gives it a real one.
#
# Why the description leads with a BRAKE rather than the capability: a tool
# described only by what it is for is read as a tool to use, and this one's cost
# is invisible from inside the model — every call parks the turn on a human and
# hands back work they delegated precisely so they would not have to do it.
# Measured over 600 local sessions the failure was not rare-and-severe but
# steady: 156 calls, and in the worst session 35 pickers on a task that had
# already been authorized in full, most of them "here is what I found, how do
# you want it handled?" — a research result reported as a question. So the
# affordance and its limit ship together, in both places a model reads about the
# tool: here (in the tools array of every request) and in the system prompt's
# fuller trigger-then-brake paragraph. Naming the legitimate triggers is
# what keeps this from reading as "never ask": the goal is fewer calls of higher
# value, not a model that pushes on through a genuinely irreversible fork.


class AskParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    questions: list[AskQuestion] = Field(
        min_length=1,
        description="The questions to ask, put to the user one screen at a time.",
    )


#: What the tool reports when the user closed the picker without choosing.
#: Deliberately NOT an error result: refusing to answer is a decision, and a
#: model that read it as a tool failure would either retry the same question or
#: stop. The last sentence is what makes the outcome actionable — a model that
#: marked an option ``recommended`` has already stated what it would do.
ASK_UNANSWERED_TEXT = (
    "The user closed the question without answering, so nothing was chosen. "
    "Do not ask again: decide yourself (take your recommended option where you "
    "gave one), then say in one line what you assumed and carry on."
)


#: What a secret question reports when the user declined or the store refused.
ASK_SECRET_NOT_PROVIDED = "<not provided>"


def _report_secret_answers(
    questions: list[AskQuestion],
    answers: dict[str, list[str]],
    context: ToolContext | None,
) -> dict[str, list[str]]:
    """Store secret answers and replace their values with the key name.

    The host's picker still returns the pasted bytes so a non-TUI embedder
    that cannot store them itself is not silently emptied. This is the last
    hop before the result is written into the transcript, so it is also the
    last hop that can keep those bytes out of the model's context.

    Also the hop that ANNOUNCES a stored credential to the model: the ask
    result names the key, but a result is one turn's text — the
    ``session_credential`` journal record (via the session hook on the
    context) is what makes the key findable on every LATER turn, the same
    way ``/credential`` does. Without it the ask path stored a secret the
    model had already forgotten the name of two turns later.
    """
    store = context.variables if context is not None else None
    store_fn = getattr(store, "store_credential", None)
    announce = getattr(context, "journal_credential", None) if context is not None else None
    reported: dict[str, list[str]] = {}
    for question in questions:
        chosen = list(answers.get(question.id, []))
        if not question.secret:
            reported[question.id] = chosen
            continue
        pasted = next((text.strip() for text in chosen if text.strip()), "")
        if not pasted or not callable(store_fn):
            reported[question.id] = [ASK_SECRET_NOT_PROVIDED]
            continue
        result = store_fn(question.id, pasted, "ask")
        credential = getattr(result, "credential", None)
        key = getattr(credential, "key", None)
        if getattr(result, "ok", False) and isinstance(key, str):
            reported[question.id] = [key]
            # Announce only on a successful store, and only the KEY: the
            # announcement is journaled into the live context and sent to the
            # provider, so the value must never ride it.
            if callable(announce):
                try:
                    announce(key, replaced=bool(getattr(result, "replaced", False)))
                except Exception:
                    # The store already succeeded; an announcement failure
                    # must not fail the ask whose answer is in hand.
                    logger.warning("could not announce stored credential", exc_info=True)
        else:
            reported[question.id] = [ASK_SECRET_NOT_PROVIDED]
    # Preserve any extra keys the host returned (should not happen) without
    # inventing values for questions we already handled.
    for key, value in answers.items():
        reported.setdefault(key, list(value))
    return reported


def _ask_report(questions: list[AskQuestion], answers: dict[str, list[str]]) -> str:
    """The answers as text the model can act on, keyed by the ids it chose.

    Each question is echoed with its answer rather than only the id: the ask
    may be several turns back by the time the model reads this, and an id on
    its own ("purge: Drop them") does not say what was agreed to.
    """
    lines: list[str] = []
    for question in questions:
        chosen = [text for text in answers.get(question.id, []) if text.strip()]
        lines.append(f"{question.id} — {question.question}")
        lines.append(f"  answer: {'; '.join(chosen) if chosen else '(not answered)'}")
    return "The user answered:\n" + "\n".join(lines)


def build_ask_tool(context: ToolContext) -> AgentTool | None:
    """CreateIf builder: the tool exists only where a human can answer it.

    Gated on the HOOK, not on ``has_ui``: a subagent inherits ``has_ui`` from
    its parent and has no human at its keyboard, so gating on the flag alone
    would let a delegated child mount a question on the parent's screen — for
    work the person watching it never asked about — and block on it. A child
    session is built without an ask handler, which makes the hook's absence the
    honest signal for every unanswerable case at once: server, exec mode,
    scheduler runs and subagents.

    ``has_ui`` is still required, and not as a duplicate of that check: a host
    declaring no UI is asserting it cannot mount a prompt, and a tool must
    believe that over a handler somebody left installed.
    """
    if context.ask_user is None or not context.has_ui:
        return None
    return AgentTool(
        name="ask",
        label="Ask",
        description=(
            "Ask the user to choose. LAST RESORT, not a checkpoint: research it, run "
            "it, or delegate it to a subagent and decide yourself, then report what you "
            "chose. Use this when the action is destructive or irreversible and the "
            "user has not EXPLICITLY approved that action, when the REQUEST ITSELF has "
            "two plausible readings that send the work in different directions and no "
            "evidence picks between them, when you need something only the user has (a "
            "credential, an access decision), or when the answer is genuinely theirs to "
            "state (a preference, a name, a roster, how they want something delivered). "
            "Two technical approaches is not ambiguity: weigh them, pick one, and say "
            "why. Work the user already asked for is authorized: do not stop to confirm "
            "it, re-ask what the conversation answered, or seek permission to continue "
            "— but that never extends to an irreversible step by implication. "
            "When you do ask, do it here INSTEAD of writing lettered options "
            "into your reply and waiting. Give each question at least two options, put "
            "the consequence of each in its description, and mark the one you recommend "
            "(it is moved to the top of the list and preselected). "
            "Every question also offers the user a free-text answer, so the options do "
            "not have to be exhaustive. Ask everything you need in ONE call: the user "
            "answers the questions back to back rather than once per turn. "
            "If you need a credential, password, or API key, set secret=true on that "
            "question (options empty, id is the env-var name). The value is stored in "
            "session memory and injected into bash; you will only ever see the key name."
        ),
        parameters=AskParams.model_json_schema(),
        # read tier: asking a question changes nothing. Gating it behind the
        # approval prompt would put one question in front of another.
        approval_tier="read",
        # Exclusive because it blocks on a HUMAN: one picker owns the keyboard,
        # and a second question mounted beside it would be unanswerable. It
        # would also hold a shared slot for as long as the user takes.
        concurrency="exclusive",
        # Interruptible because this call is parked on a HUMAN, and nothing else
        # can settle it. A non-interruptible tool is cancelled by nothing but
        # its own return (the approval gate documents that failure: an abort
        # landed while a turn sat on the prompt and the runner went on waiting),
        # so a stop or a steering message arriving while the question is up has
        # to be able to end the call — the loop's steering poll cancels it and
        # the host takes the picker off screen. Esc remains how the user
        # DECLINES to answer: that returns a result, and is not a cancellation.
        interruptible=True,
        execute=execute_ask,
    )


@_guard("ask")
async def execute_ask(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Put the questions to the user and report what they chose."""
    try:
        params = AskParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "ask", exc)
    ask_user = context.ask_user if context else None
    if ask_user is None:
        # Unreachable through the advertised tool (the builder above refuses to
        # create it without a hook), so this is a host wiring fault rather than
        # a user decision — and it must not be reported as one, or the model
        # would "fall back to its recommendation" on a session where the user
        # was never shown anything.
        return _error(
            tool_call_id,
            "ask",
            "No interactive surface is attached to this session, so the user cannot "
            "be asked. Decide without them.",
        )
    answers = await ask_user(params.questions)
    if not answers or not any(any(text.strip() for text in chosen) for chosen in answers.values()):
        # One outcome, whether the host answered None (escaped) or handed back
        # a mapping with nothing in it (confirmed an empty multi-select): the
        # user chose nothing, and splitting that into two results would give the
        # model a distinction it cannot act on differently.
        return _text(tool_call_id, "ask", ASK_UNANSWERED_TEXT)
    # Secret answers are stored by the host and reported as the KEY NAME
    # only. The raw value must never ride the tool result: that text is
    # persisted to the transcript and replayed to the provider.
    reported = _report_secret_answers(params.questions, answers, context)
    return _text(
        tool_call_id,
        "ask",
        _ask_report(params.questions, reported),
        details={"answers": {key: list(value) for key, value in reported.items()}},
    )
