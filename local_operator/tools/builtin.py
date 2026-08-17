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

import asyncio
import base64
import contextlib
import difflib
import fnmatch
import io
import json
import mimetypes
import os
import re
import signal as signal_module
import time
import traceback
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from local_operator.harness.approval import ask_approval
from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    AgentToolUpdate,
    ApprovalDescribeFn,
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
from local_operator.helpers import heif_image_module, pillow_image_module
from local_operator.media import ImageInfo, sniff_image_file
from local_operator.optional import missing_extra_error
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
#: Refuse to decode above this pixel count (~200 MB of RGBA at 4 bytes/pixel).
#: Checked against the header dimensions BEFORE the decode allocates, because
#: a decompression bomb is small on disk by construction: the byte cap above
#: cannot see it coming. 50M pixels is ~7000x7000, comfortably above any
#: camera or display this reads from and comfortably below Pillow's own 89M
#: bomb threshold, so the refusal is ours and is an error rather than a warning.
READ_IMAGE_MAX_PIXELS = 50_000_000
#: Long-edge ceiling in pixels for an image handed to the model. Anthropic
#: downsizes anything above 1568 server-side and bills the resized token count
#: either way, so pixels past this line are pure upload with zero fidelity
#: reaching the model; omp uses the same number and this repo's snapcompact
#: already renders its frames at 1568. Measured on real files: a 2560x1600 UI
#: screenshot costs 5,461 image tokens untouched and 2,049 at 1568 (2.7x), and
#: a 4032x3024 phone photo — 1.5 MB as JPEG, so it passes the byte cap easily
#: — costs 16,257 tokens untouched against 2,459 resized (6.6x).
READ_IMAGE_MAX_EDGE = 1568
#: Encoded-byte threshold for the image block, before base64 inflates it by
#: 4/3. Two jobs. It decides when a small in-bounds image is forwarded
#: VERBATIM (cheapest and lossless — no re-encode can improve an image the
#: model will see at its original size), and it decides when the resized PNG
#: is too fat to keep, which is the only reason lossy JPEG is ever reached.
#:
#: A TRIGGER, not a guarantee: the ladder stops after JPEG rather than
#: chasing quality down, so pathological input still lands above it (uniform
#: noise at 1568x1176 measured 1.19 MiB of quality-85 JPEG). The guarantee is
#: the wall this is set against — Anthropic rejects images over 5 MB of
#: base64, and the long-edge cap means even that pathological case encodes to
#: 1.59 MiB, 32% of the wall. 1 MiB was picked to leave 3.7x headroom on the
#: ordinary path, and the fat cases are real: a photographic 1568x1176 frame
#: re-encodes to a 3.3 MiB PNG (4.46 MiB base64, inside 5 MB by only 12%) and
#: to a 804 KiB JPEG.
READ_IMAGE_MAX_BYTES = 1024 * 1024
#: JPEG quality used for that fallback. 85 is the standard visually-lossless
#: point; on the sampled files it turned 1.9 MB of re-encoded PNG into 271 KB.
READ_IMAGE_JPEG_QUALITY = 85
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

    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []

    async def _pump(stream: asyncio.StreamReader | None, sink: list[bytes]) -> None:
        # Both pipes were requested at spawn, so neither is ever None here;
        # the guard keeps the reader honest instead of asserting.
        if stream is None:
            return
        try:
            while True:
                chunk = await stream.read(65536)
                if not chunk:
                    break
                sink.append(chunk)
        except (ConnectionResetError, BrokenPipeError):
            pass

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
        stdout = b"".join(stdout_chunks).decode("utf-8", errors="replace")
        stderr = b"".join(stderr_chunks).decode("utf-8", errors="replace")
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

    while True:
        waiters: list[asyncio.Task[object]] = [wait_task, stdout_task, stderr_task]
        if abort_waiter is not None:
            waiters.append(abort_waiter)
        if wait_task.done():
            break  # finished already — never misreport as timeout
        remaining = deadline - loop.time()
        if remaining <= 0:
            timed_out = True
            _kill()
            break
        done, _pending = await asyncio.wait(waiters, timeout=min(0.25, remaining))
        if wait_task in done:
            break
        if abort_waiter is not None and abort_waiter in done:
            aborted = True
            _kill()
            break
        if loop.time() >= next_update:
            _emit_update()
            next_update = loop.time() + 0.5

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

    if abort_waiter is not None and not abort_waiter.done():
        abort_waiter.cancel()
        with contextlib.suppress(BaseException):
            await abort_waiter

    if aborted:
        partial = _bash_output_summary(
            b"".join(stdout_chunks).decode("utf-8", errors="replace"),
            b"".join(stderr_chunks).decode("utf-8", errors="replace"),
        )
        return _error(
            tool_call_id,
            "bash",
            f"aborted ({(signal.reason or 'aborted') if signal else 'aborted'}): "
            f"{params.command}\n{partial}",
        )

    stdout_raw = b"".join(stdout_chunks).decode("utf-8", errors="replace")
    stderr_raw = b"".join(stderr_chunks).decode("utf-8", errors="replace")
    return_code = process.returncode if process.returncode is not None else -1

    # Both streams may end up carrying a marker, so reserve room for two.
    budget = TOOL_OUTPUT_LIMIT_CHARS - 2 * len(BASH_TRUNCATION_MARKER)
    spill_details: dict[str, Any] | None = None
    if len(stdout_raw) + len(stderr_raw) > budget:
        # ONE spill for the WHOLE transcript, in exactly the framing the model
        # already sees. Spilling the two streams separately would hand out two
        # handles for one command and make "line 900" ambiguous; spilling a
        # differently-framed copy would make the footer's line numbers point
        # somewhere other than where they resolve.
        combined = _bash_output_summary(stdout_raw, stderr_raw)
        meta = _spill(combined, "bash", context)
        stdout_budget, stderr_budget = _stream_budgets(
            stdout_raw, stderr_raw, budget, failed=(return_code != 0 or timed_out)
        )
        footer = ""
        if meta is None:
            stdout = truncate_output(stdout_raw, stdout_budget)
            stderr = truncate_output(stderr_raw, stderr_budget)
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
            failed = return_code != 0 or timed_out
            suggested = (stderr_span if failed and stderr_span else None) or stdout_span
            footer = _spill_footer(meta, suggested)
    else:
        stdout, stderr = stdout_raw, stderr_raw
        footer = ""

    parts = [f"exit code: {return_code}", _bash_output_summary(stdout, stderr)]
    if timed_out:
        parts.insert(0, f"TIMEOUT after {params.timeout}s (process killed)")
    return _text(tool_call_id, "bash", "\n".join(parts) + footer, details=spill_details)


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
            "internal URL: skill://<name>, or spill://<id> to expand an output "
            "that was truncated (append '?q=<regex>' to search inside it "
            "instead of paging through it)."
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


def _forward_image_undecoded(data: bytes, info: ImageInfo) -> tuple[bytes, str, str]:
    """Ship image bytes VERBATIM on a host with no usable Pillow.

    Pillow reaches a default install as a pillow-heif dependency rather than a
    direct one, and pillow-heif is the most platform-fragile wheel this
    project pulls. When it is missing or broken there is no decoder, so there
    is also no resize and no validation beyond the header — the two things
    :func:`_encode_image_for_model` normally provides. Refusing every image on
    such a host would be the worse trade: a screenshot the model can look at
    beats a paragraph explaining why it cannot, and the format is one the
    provider clients already serialize.

    What remains enforceable from the header alone is the BYTE cap, so that is
    the line. Above it the answer is an error, because forwarding an unbounded
    blob is how a session ends up wedged behind a provider that refuses it.
    """
    if not info.sendable:
        # Not a degrade: no provider accepts HEIC, so forwarding it verbatim
        # would GUARANTEE the refusal rather than risk it. Transcoding is the
        # only way to send one, and transcoding is what is unavailable.
        raise ValueError(missing_extra_error("images", "HEIC/HEIF decoding"))
    if len(data) > READ_IMAGE_MAX_BYTES:
        raise ValueError(
            f"it is {len(data)} bytes, over the {READ_IMAGE_MAX_BYTES}-byte cap for an "
            f"unresized image, and {missing_extra_error('images', 'resizing it')}"
        )
    summary = f"{info.mime_type}, {info.dimensions or 'dimensions unknown'}, {len(data)} bytes"
    if info.width and info.height and max(info.width, info.height) > READ_IMAGE_MAX_EDGE:
        # Worth saying rather than swallowing: this is the case the resize
        # exists for, and the model is about to be billed several times the
        # tokens for it. Naming it is also the only hint anyone gets that the
        # host is missing the extra.
        summary += ", too large to send efficiently and no decoder to resize it"
    else:
        summary += ", forwarded without resizing"
    return data, info.mime_type, summary


def _guard_pixel_budget(width: int, height: int) -> None:
    """Refuse an image whose pixel count would dominate the process.

    A decompression bomb is small on disk by construction, so the byte cap
    cannot see it coming and only the dimensions can.
    """
    if width * height > READ_IMAGE_MAX_PIXELS:
        raise ValueError(
            f"it is {width}x{height} ({width * height:,} pixels) and the decode limit is "
            f"{READ_IMAGE_MAX_PIXELS:,} pixels"
        )


def _encode_image_for_model(data: bytes, info: ImageInfo) -> tuple[bytes, str, str]:
    """Decode, bound and re-encode image bytes for a provider.

    Returns ``(payload, wire_mime, summary)``; raises ``ValueError`` with a
    model-readable message when the bytes will not decode. The raise is
    load-bearing: a corrupt or truncated image forwarded as an image block
    earns a ``Could not process image`` 400 from Anthropic. The session layer
    now recovers from that, but a backstop is not a licence — the bad block is
    still a wasted round trip and a degraded session, and decoding here is
    where it is cheap to avoid. So the decode is never skipped, not even on
    the verbatim path.

    The ladder, cheapest first:

    1. Verbatim, when the image is already inside both bounds and in a format
       the clients serialize. No re-encode can improve an image the model sees
       at its original size, and PNG round-tripping routinely makes files
       BIGGER (a 2560x1600 UI screenshot measured 550 KB on disk against
       335 KB re-encoded only because the resize came with it).
    2. Resize to :data:`READ_IMAGE_MAX_EDGE` and re-encode as PNG. Lossless,
       which is what a screenshot of 9-pixel text needs.
    3. JPEG when that PNG blows :data:`READ_IMAGE_MAX_BYTES`, and only when it
       actually comes out smaller. PNG is a bad photographic codec and that is
       the usual case here — the sampled 1672x941 photographic PNG re-encoded
       to 1.9 MB of PNG against 271 KB of quality-85 JPEG — but it is not the
       only one, so the choice is measured rather than assumed.

    With no decoder available the whole ladder collapses to
    :func:`_forward_image_undecoded`.
    """
    image_module = pillow_image_module() if info.sendable else heif_image_module()
    if image_module is None:
        return _forward_image_undecoded(data, info)

    # The pixel cap wants to fire BEFORE the decode allocates, and for every
    # format except HEIF the header already answers it. HEIF keeps its size in
    # a meta-nested ispe box that media.sniff_image deliberately does not walk,
    # so those are capped below on the decoded size instead — later than ideal,
    # but the only point at which the number exists.
    if info.width and info.height:
        _guard_pixel_budget(info.width, info.height)

    try:
        image = image_module.open(io.BytesIO(data))
        width, height = image.size
        _guard_pixel_budget(width, height)
        # Multi-frame sources never pass through: providers read frame 0 and
        # ignore the rest, so an animation's other frames are bytes uploaded
        # to be discarded.
        frames = getattr(image, "n_frames", 1)
        image.load()

        long_edge = max(width, height)
        if (
            info.sendable
            and frames == 1
            and long_edge <= READ_IMAGE_MAX_EDGE
            and len(data) <= READ_IMAGE_MAX_BYTES
        ):
            return data, info.mime_type, f"{info.mime_type}, {width}x{height}, {len(data)} bytes"

        if long_edge > READ_IMAGE_MAX_EDGE:
            scale = READ_IMAGE_MAX_EDGE / long_edge
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            image = image.resize(size, image_module.LANCZOS)

        # Palette and high-bit-depth modes are legal PNG but not legal JPEG,
        # and rung 3 must not be the first place a mode problem shows up.
        image = image.convert("RGBA" if image.mode in ("RGBA", "LA", "PA", "P") else "RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        payload, wire_mime = buffer.getvalue(), "image/png"

        if len(payload) > READ_IMAGE_MAX_BYTES:
            if image.mode == "RGBA":
                # JPEG has no alpha channel. Compositing onto white rather
                # than dropping the channel keeps a transparent-background
                # diagram legible instead of rendering it onto black.
                flat = image_module.new("RGB", image.size, (255, 255, 255))
                flat.paste(image, mask=image.getchannel("A"))
                image = flat
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=READ_IMAGE_JPEG_QUALITY)
            jpeg = buffer.getvalue()
            # Take the smaller of the two, so the lossy rung can never make the
            # result WORSE on both axes at once. PNG beats JPEG on flat
            # synthetic images, and one of those clearing the budget is
            # possible even though nothing sampled here did it.
            if len(jpeg) < len(payload):
                payload, wire_mime = jpeg, "image/jpeg"
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 — Pillow raises OSError, SyntaxError and its own
        raise ValueError(f"could not decode the image data ({type(exc).__name__}: {exc})") from exc

    summary = f"{wire_mime}, {image.width}x{image.height}, {len(payload)} bytes"
    if image.size != (width, height) or wire_mime != info.mime_type:
        # State the source whenever what the model sees is not what is on
        # disk. Otherwise a model comparing this against `ls -l` output, or
        # against a later re-read, has no way to reconcile the two.
        summary += f"; source {width}x{height} {info.mime_type}"
    return payload, wire_mime, summary


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
        return _text(tool_call_id, "read", content, details={"url": target})

    cwd = _safe_cwd(context)
    path, inside, resolvable = _resolve_workspace_path(target, cwd)
    if not path.exists():
        return _error(tool_call_id, "read", f"Path does not exist: {path}")

    # Outside-workspace reads escalate to an approval prompt regardless of
    # the read tier auto-approval the host normally applies.
    if not inside:
        description = _approval_description(path, inside, "read", resolvable)
        if not await _check_approval(context, "read", description):
            return _error(tool_call_id, "read", "User declined to read this file.")

    if path.is_dir():
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())
        return _text(
            tool_call_id,
            "read",
            f"Directory listing of {path} ({len(entries)} entries):\n" + "\n".join(entries),
            details={"path": str(path)},
        )

    # Stat and SNIFF before reading the body: an oversized file is refused
    # instead of loaded, and the applicable ceiling depends on what the file
    # is. Classification is by CONTENT, never by extension — a `.png` holding
    # an HTML error page must not reach a provider as an image, and a
    # screenshot saved with no extension at all is still a screenshot.
    # `sniff_image_file` reads at most 64 KB and never imports a decoder, so a
    # text read pays a short header read to learn it is a text read.
    size = path.stat().st_size
    info = sniff_image_file(str(path))
    limit = READ_IMAGE_LIMIT_BYTES if info else READ_FILE_LIMIT_BYTES
    if size > limit:
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

    data = path.read_bytes()

    if info:
        try:
            payload, wire_mime, summary = _encode_image_for_model(data, info)
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

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()

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
            "PNG/JPEG/GIF/WebP/HEIC files come back as a viewable image."
        ),
        parameters=ReadParams.model_json_schema(),
        approval_tier="read",
        # read model: parallel reads are the common batch shape.
        concurrency="shared",
        interruptible=False,
        execute=execute_read,
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
    path.write_text(params.content, encoding="utf-8")
    verb = "Overwrote" if existed else "Created"
    details = _diff_details(str(path), previous, params.content)
    return _text(
        tool_call_id,
        "write",
        f"{verb} {path} ({len(params.content)} chars).",
        details=details,
    )


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
    )


# ---------------------------------------------------------------------------
# edit
# ---------------------------------------------------------------------------


class EditParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="File to edit.")
    old_text: str = Field(description="Exact text to find (must match verbatim).")
    new_text: str = Field(description="Replacement text.")
    replace_all: bool = Field(
        default=False,
        description="Replace every occurrence instead of requiring exactly one.",
    )


@_guard("edit")
async def execute_edit(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Exact-match string replacement in a file.

    Ambiguity is an error, not a guess: with ``old_text`` matching more than
    once and ``replace_all`` unset the tool refuses, because silently editing
    the first occurrence is how edits corrupt the wrong site.
    """
    try:
        params = EditParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "edit", exc)
    if not params.path.strip():
        return _error(tool_call_id, "edit", "path must be a non-empty string")
    if params.old_text == "":
        return _error(tool_call_id, "edit", "old_text must be a non-empty string")

    path, inside, _resolvable = _resolve_workspace_path(params.path, _safe_cwd(context))
    if not path.is_file():
        return _error(tool_call_id, "edit", f"File does not exist: {path}")

    content = path.read_text(encoding="utf-8")
    occurrences = content.count(params.old_text)
    if occurrences == 0:
        return _error(
            tool_call_id,
            "edit",
            "old_text not found in the file. Re-read the file to get the exact "
            "current text (whitespace included) and retry.",
        )
    if occurrences > 1 and not params.replace_all:
        return _error(
            tool_call_id,
            "edit",
            f"old_text matches {occurrences} places; include more surrounding "
            "context to make it unique, or set replace_all=true.",
        )
    # Write-tier approval is the loop's gate; see execute_bash.

    if params.replace_all:
        updated = content.replace(params.old_text, params.new_text)
    else:
        updated = content.replace(params.old_text, params.new_text, 1)
    path.write_text(updated, encoding="utf-8")
    replaced = occurrences if params.replace_all else 1
    details = _diff_details(str(path), content, updated)
    return _text(
        tool_call_id,
        "edit",
        f"Edited {path}: replaced {replaced} occurrence(s) of old_text.",
        details=details,
    )


def build_edit_tool() -> AgentTool:
    return AgentTool(
        name="edit",
        label="Edit",
        describe_approval=_describe_path_approval("edit"),
        description="Replace exact text in a file (errors on missing or ambiguous matches).",
        parameters=EditParams.model_json_schema(),
        approval_tier="write",
        # edit model: two concurrent edits on one file corrupt each
        # other's match anchors; exclusive serializes the read-modify-write.
        concurrency="exclusive",
        interruptible=False,
        execute=execute_edit,
    )


# ---------------------------------------------------------------------------
# glob
# ---------------------------------------------------------------------------


class GlobParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: str = Field(
        description="Glob pattern relative to the working directory ('**/*.py' supported)."
    )


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
        return _error(
            tool_call_id,
            "glob",
            "pattern must be a relative glob within the working directory "
            "(no absolute paths, no '..').",
        )

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
    # it did not exist; now the whole list is one range read away.
    total = len(matches)
    full = "\n".join(matches)
    shown = matches[:GLOB_RESULT_LIMIT]
    body, spill_details = _capped_list_body(full, "\n".join(shown), "glob", context)
    header = f"{len(shown)} match(es) for '{params.pattern}'"
    if total > len(shown):
        header += f" of {total} (capped at {GLOB_RESULT_LIMIT})"
    return _text(tool_call_id, "glob", header + ":\n" + body, details=spill_details)


def build_glob_tool() -> AgentTool:
    return AgentTool(
        name="glob",
        label="Glob",
        description="Find files and directories by glob pattern ('**' supported).",
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


def _glob_walk(root: Path, pattern: str) -> list[str]:
    """The walk half of execute_glob, run in a worker thread."""
    return sorted(
        p.relative_to(root).as_posix() + ("/" if p.is_dir() else "") for p in root.glob(pattern)
    )


def _glob_matches(rel_path: str, pattern: str) -> bool:
    """Match ``rel_path`` against ``pattern`` (basename fallback for bare globs)."""
    if fnmatch.fnmatch(rel_path, pattern):
        return True
    name = rel_path.rsplit("/", 1)[-1]
    return fnmatch.fnmatch(name, pattern)


def _walk_files(root: Path) -> list[Path]:
    """Walk ``root`` depth-first, pruning VCS/vendor/build trees and every
    dotdir (.git history and node_modules are noise the model never wants)."""
    files: list[Path] = []

    def _walk(directory: Path) -> None:
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            return
        for entry in entries:
            if entry.is_symlink():
                continue  # never follow links: cycles and out-of-tree escapes
            if entry.is_dir():
                if entry.name in _GREP_PRUNE_DIRS or entry.name.startswith("."):
                    continue
                _walk(entry)
            elif entry.is_file():
                files.append(entry)

    _walk(root)
    return files


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


def _grep_scan(
    files: list[Path],
    base: Path,
    regex: re.Pattern[str],
    include: str | None,
) -> tuple[list[str], int, int]:
    """The filesystem+regex half of execute_grep, run in a worker thread.

    Returns ``(matches, files_searched, files_skipped)``. Kept synchronous and
    self-contained so ``asyncio.to_thread`` can carry it off the event loop;
    the deadline bounds a backtracking pattern without touching the loop.
    """
    deadline = time.monotonic() + GREP_SCAN_DEADLINE_S
    matches: list[str] = []
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
        for lineno, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                matches.append(f"{rel}:{lineno}:{line}")
                if len(matches) >= GREP_SPILL_MATCH_LIMIT:
                    break
        if len(matches) >= GREP_SPILL_MATCH_LIMIT:
            break
    return matches, files_searched, files_skipped


@_guard("grep")
async def execute_grep(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Regex search over files; ripgrep-free pure-Python implementation."""
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

    if target.is_file():
        files: list[Path] = [target]
        base = target.parent
    else:
        base = target
        files = _walk_files(target)

    # The scan is FILESYSTEM + REGEX work on model-controlled input; running
    # it on the event loop would pin the CPU on a backtracking pattern or a
    # large tree and make Ctrl+C unprocessable. It runs in a worker thread
    # raced against the abort signal, with a wall-clock cap bounding the
    # pathological-regex case (regexes are not classified).
    scan_result, aborted = await _run_with_abort(
        asyncio.to_thread(_grep_scan, files, base, regex, params.include),
        signal,
        lambda: None,
    )
    if aborted:
        return _error(tool_call_id, "grep", "Search aborted.")
    matches, files_searched, files_skipped = scan_result

    if not matches:
        skipped_note = (
            f" ({files_skipped} file(s) skipped over the 1MB cap)" if files_skipped else ""
        )
        return _text(
            tool_call_id,
            "grep",
            f"No matches for '{params.pattern}' in {files_searched} " f"file(s){skipped_note}.",
            useless=True,
            details={"useless": True},
        )
    total = len(matches)
    shown = matches[:GREP_MATCH_LIMIT]
    body, spill_details = _capped_list_body("\n".join(matches), "\n".join(shown), "grep", context)
    header = f"{len(shown)} match(es) for '{params.pattern}'"
    if total > len(shown):
        header += f" of {total}{'+' if total >= GREP_SPILL_MATCH_LIMIT else ''}"
        header += f" (showing first {GREP_MATCH_LIMIT})"
    if files_skipped:
        header += f" ({files_skipped} file(s) skipped over the 1MB cap)"
    return _text(tool_call_id, "grep", header + ":\n" + body, details=spill_details)


def build_grep_tool() -> AgentTool:
    return AgentTool(
        name="grep",
        label="Grep",
        description="Regex search across files, returning 'path:line:text' matches.",
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

#: In-memory todo lists keyed by NON-EMPTY session id. The host may attach a
#: durable store to the ToolContext (``todos`` dict) — we prefer that so
#: transcripts can replay todo state — but a bare context still works via
#: this table (keyed by the context object's id when no session id exists).
TODO_STORE: dict[str, list[dict[str, str]]] = {}
#: Fallback store for contexts without a session id, so their lists never
#: collide under the shared "" key. Keyed by the context object's id rendered
#: as a string, so every todo store in this module has one key type.
_CONTEXT_TODO_STORE: dict[str, list[dict[str, str]]] = {}


class TodoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["init", "done", "view"] = Field(
        description="init: set the list; done: mark one item done; view: show the list."
    )
    items: list[str] = Field(
        default_factory=list,
        description="Todo texts (required for 'init', item text for 'done').",
    )


#: Every todo store — host-attached or module-level — maps one owner key to
#: that owner's list of ``{"text": ..., "status": ...}`` items.
TodoStore = dict[str, list[dict[str, str]]]


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

    if params.op == "init":
        if not params.items:
            return _error(tool_call_id, "todo", "'init' requires a non-empty items list")
        store[key] = [{"text": item, "status": "pending"} for item in params.items]
        return _text(
            tool_call_id,
            "todo",
            f"Todo list initialized with {len(params.items)} item(s).",
        )

    current = store.get(key, [])
    if params.op == "done":
        if not params.items:
            return _error(tool_call_id, "todo", "'done' requires items with the item text")
        target = params.items[0]
        for item in current:
            if item["text"] == target and item["status"] != "done":
                item["status"] = "done"
                done = sum(1 for i in current if i["status"] == "done")
                return _text(
                    tool_call_id,
                    "todo",
                    f"Marked done: {target} ({done}/{len(current)} complete).",
                )
        return _error(
            tool_call_id,
            "todo",
            f"No pending todo matching '{target}'. Use todo view to see current items.",
        )

    # op == "view"
    if not current:
        return _text(
            tool_call_id,
            "todo",
            "No todos recorded yet.",
            useless=True,
            details={"useless": True},
        )
    marks = {"done": "x", "pending": " "}
    lines = [f"- [{marks.get(item['status'], ' ')}] {item['text']}" for item in current]
    return _text(tool_call_id, "todo", "\n".join(lines))


def build_todo_tool() -> AgentTool:
    return AgentTool(
        name="todo",
        label="Todo",
        description="Track a visible task list (init / done / view).",
        parameters=TodoParams.model_json_schema(),
        # read tier exemption: todo mutates only session-local bookkeeping
        # (no files, no autonomous turns), so it stays auto-approved.
        approval_tier="read",
        # init rewrites the whole list; concurrent calls would lose one,
        # so the tool runs exclusive despite being cheap.
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
        description="Delay before first fire: '45s'|'30m'|'2h'|'7d'|'1w'.",
    )
    at: str | None = Field(
        default=None,
        description="First fire time: 'HH:MM', '+<duration>', or ISO datetime.",
    )
    every: str | None = Field(default=None, description="Repeat interval duration, e.g. '1h'.")
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
        description="Schedule a future wake (create/list/cancel), e.g. 'in 30m'.",
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
    names = _variable_store(context).names()
    shown = names if len(names) <= 100 else names[:100] + ["…"]
    body = "\n".join(shown) if shown else "(no variables defined)"
    return _text(
        tool_call_id,
        "list_variables",
        f"{len(names)} variable(s) available:\n{body}",
        details={"count": len(names)},
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
)

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
        description="open (start a surface at a URL) | goto | read (page text) | "
        "snapshot (accessibility tree with click refs) | screenshot | click | "
        "type | close."
    )
    url: str = Field(default="", description="http(s) URL for 'open'/'goto'.")
    path: str = Field(default="", description="Destination file for 'screenshot'.")
    selector: str = Field(
        default="",
        description="CSS selector or a snapshot ref (e5) for 'click'/'type'; "
        "scopes the text for 'read' (default: body).",
    )
    text: str = Field(default="", description="Text to enter for 'type'.")


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
    """
    try:
        return _cmux_binary() is not None
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
    if action in ("open", "goto"):
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
        f"Opened browser surface {surface_id}: {_page_line(title, href)}",
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
    """Drive the CMUX browser. Degrades to a clear error when cmux is absent."""
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
    if not cmux_browser_available():
        return _error(
            tool_call_id,
            "browser",
            "CMUX browser not available: no cmux binary on PATH and no "
            "CMUX_BUNDLED_CLI_PATH. This host cannot drive a browser.",
        )
    # Before the state lookup and before every subprocess, including the
    # liveness probe below: see _validate_browser_args.
    problem = _validate_browser_args(action, params)
    if problem:
        return _error(tool_call_id, "browser", problem)
    state = _browser_state(context)

    # 'open' creates the surface and 'close' must stay callable without one, so
    # both run before the have-a-surface gate below.
    if action == "open":
        return await _browser_open(tool_call_id, state, params.url)
    if action == "close":
        return await _browser_close(tool_call_id, state)
    if not state.surface_id:
        return _error(tool_call_id, "browser", "no browser surface open — use 'open' first")
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
    """Advertise the browser tool only when a CMUX browser is reachable.

    Mirrors the wake builder: an environment-specific capability that returns
    None (excluded from the inventory) when the host cannot support it.

    There is deliberately no headless fallback. This repo ships no browser
    engine — playwright belongs to the pre-rewrite codebase and appears in no
    dependency group — and pulling one into the default install would add a
    ~150 MB browser download to a dependency set that is kept small on
    purpose. A host without cmux therefore has no browser tool at all, which
    is honest, and the agent still reaches static pages through `bash` and
    curl.
    """
    if not cmux_browser_available():
        return None
    return AgentTool(
        name="browser",
        label="Browser",
        describe_approval=_describe_browser_approval,
        description=(
            "Drive the CMUX browser: open/goto a URL, read page text, snapshot "
            "the accessibility tree for click refs, click, type, screenshot, close."
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


class TaskParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(description="Short label for the subagent (shown in the jobs list).")
    prompt: str = Field(
        description="The full instructions the subagent runs in a fresh child session."
    )


class WaitParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(description="Job id returned by the 'task' tool (or listed by 'jobs').")
    wait_ms: int = Field(
        default=30_000,
        gt=0,
        le=300_000,
        description="Max ms to block for the job to settle (capped at 300000).",
    )


class JobsParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


#: Formatted status text shared by ``wait``'s settled return and its detail
#: payload.  A child report can dwarf an ordinary tool result, so it uses the
#: same bounded, lossless spill path as every other verbose tool.
def _job_summary(job: Any, context: ToolContext | None = None) -> tuple[str, dict[str, Any] | None]:
    """Return a context-bounded handoff while keeping the full report readable."""
    text = f"job {job.id} ({job.label}) [{job.status}]"
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
    """Launch a one-shot subagent as a background job; return its job id."""
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
    try:
        job_id = launcher(params.label, params.prompt)
    except Exception as exc:  # noqa: BLE001 — engine failure surfaces as an error result
        return _error(tool_call_id, "task", f"could not launch subagent: {exc}")
    return _text(
        tool_call_id,
        "task",
        f"launched subagent job {job_id} ({params.label}); use 'wait' with "
        f"job_id={job_id} to await it, or 'jobs' to list running work.",
        details={"job_id": job_id, "label": params.label},
    )


def build_task_tool(context: ToolContext) -> AgentTool | None:
    if context.subagent_launcher is None:
        return None
    return AgentTool(
        name="task",
        label="Subagent task",
        describe_approval=_describe_task_approval,
        description=(
            "Launch a one-shot subagent in a fresh child session, running "
            "in the background; returns a job id to await with 'wait'."
        ),
        parameters=TaskParams.model_json_schema(),
        # Spawns autonomous child work, so it rides the write gate just like
        # scheduling a wake: the user approves starting the child.
        approval_tier="write",
        concurrency="exclusive",
        interruptible=False,
        execute=execute_task,
    )


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
    job = jobs.get(params.job_id)
    if job is None:
        return _error(tool_call_id, "wait", f"unknown job {params.job_id}")

    deadline = time.monotonic() + params.wait_ms / 1000.0
    while job.status == "running":
        if signal is not None and signal.aborted:
            return _text(
                tool_call_id,
                "wait",
                f"job {params.job_id} still running (wait aborted)",
                details={"job_id": params.job_id, "status": "running"},
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return _text(
                tool_call_id,
                "wait",
                f"job {params.job_id} still running after {params.wait_ms}ms",
                details={"job_id": params.job_id, "status": "running"},
            )
        await asyncio.sleep(min(0.05, remaining))
        job = jobs.get(params.job_id)
        if job is None:
            return _error(tool_call_id, "wait", f"job {params.job_id} disappeared")
    text, spill_details = _job_summary(job, context)
    details = {"job_id": job.id, "status": job.status}
    if spill_details:
        details.update(spill_details)
    return _text(
        tool_call_id,
        "wait",
        text,
        details=details,
    )


def build_wait_tool(context: ToolContext) -> AgentTool | None:
    if context.jobs is None:
        return None
    return AgentTool(
        name="wait",
        label="Wait for job",
        description=(
            "Block up to wait_ms for a background job to settle, returning its "
            "final output/status. Times out if still running."
        ),
        parameters=WaitParams.model_json_schema(),
        # read-only observation of job state; blocks the turn but changes nothing.
        approval_tier="read",
        concurrency="exclusive",
        interruptible=True,
        execute=execute_wait,
    )


@_guard("jobs")
async def execute_jobs(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """List running and recently-settled background jobs."""
    try:
        JobsParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "jobs", exc)

    jobs = context.jobs if context else None
    if jobs is None:
        return _error(
            tool_call_id,
            "jobs",
            "job tracking is not available in this session (no job manager attached).",
        )
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
        # ago were indistinguishable. ``start_time`` is the age's only source;
        # it is guarded because a replayed or embedder-supplied row may carry
        # none, and a missing clock is worth 0.0s rather than a traceback.
        reference = (
            job.settled_at
            if job.status != "running" and job.settled_at
            else getattr(job, "start_time", None)
        )
        elapsed = max(now - reference, 0.0) if reference else 0.0
        lines.append(f"{job.id}  {job.status:<9}  {elapsed:6.1f}s  {job.label}")
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
            "List running and recently-settled background jobs (task/bash) "
            "with their id, status and elapsed time."
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
# ``build_hub_tool``). ONE tool rather than five (send/ask/steer/cancel/resume)
# because they share a target and a body and differ only in intent — five
# entries would spend five tool-schema slots and five descriptions on one
# concept, and the model would still have to learn which of them means "and
# wait for the answer". Named ``hub`` after the surface the same ops have in
# omp, whose shape this follows deliberately: ``to`` addresses one peer or
# ``"all"``, delivery returns per-recipient receipts, and asking is a send
# that waits.
#
# The two shapes are not cosmetic. A parent may address, redirect, stop and
# resume its children; a child has exactly one peer (its parent) and no
# children of its own, so it gets a tool with no ``op`` and no ``to`` at all.
# Advertising the parent schema to a child would spend the child's context on
# four ops it cannot use and invite it to try them.


class HubParams(BaseModel):
    """Parent-side hub arguments."""

    model_config = ConfigDict(extra="forbid")

    op: Literal["send", "ask", "steer", "cancel", "resume"] = Field(
        description=(
            "send: a note, no reply waited for. ask: a question, blocks for the "
            "subagent's answer. steer: change what it is doing (becomes part of its "
            "instructions). cancel: stop it. resume: relaunch a stopped subagent "
            "against its own transcript so it continues where it left off."
        )
    )
    # A plain array, NOT ``str | list[str]``: pydantic renders a union as
    # ``anyOf``, and this module's schemas reach Gemini verbatim as
    # ``function_declarations`` (providers/clients.py builds the body with
    # ``tool.parameters`` untouched). A construct one provider rejects would
    # fail every request in the session, not just the hub call — no builtin
    # here uses a non-nullable anyOf, and this is not the tool to be first.
    to: list[str] = Field(
        min_length=1,
        description=(
            "Who to address: job ids from 'task'/'jobs', subagent labels, or "
            '["all"] for every running subagent. Several ids address several '
            "subagents. 'ask' and 'resume' take exactly one."
        ),
    )
    message: str | None = Field(
        default=None,
        description=(
            "The body. Required for send/ask/steer, and for resume (what to do next); "
            "ignored by cancel."
        ),
    )
    timeout_ms: int = Field(
        default=120_000,
        gt=0,
        le=600_000,
        description="op='ask' only: how long to wait for the answer.",
    )


class HubChildParams(BaseModel):
    """Child-side hub arguments: one peer, one direction, no ops."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(
        description="What to tell the parent agent. Answers its question when it asked one."
    )


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
    body = " ".join(str(args.get("message") or "").split())
    if len(body) > 60:
        body = body[:57] + "..."
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

    if params.op != "cancel" and not (params.message or "").strip():
        return _error(tool_call_id, "hub", f"op='{params.op}' needs a message.")

    ids, errors = _hub_targets(comms, params.to)
    if not ids:
        return _error(
            tool_call_id,
            "hub",
            "; ".join(errors) or "no subagent matched; use 'jobs' to list them.",
        )
    # A question and a resume both have exactly one answer, so they refuse a
    # fan-out rather than silently acting on the first match.
    if params.op in ("ask", "resume") and len(ids) > 1:
        return _error(
            tool_call_id,
            "hub",
            f"op='{params.op}' addresses one subagent at a time; got {len(ids)}.",
        )

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
                "it is still running and the question is in its context.",
                details={"op": "ask", "job_id": reply.job_id, "timed_out": True},
            )
        return _text(
            tool_call_id,
            "hub",
            f"{reply.label} ({reply.job_id}) replied:\n{reply.text}",
            details={"op": "ask", "job_id": reply.job_id, "reply": reply.text},
        )

    if params.op == "resume":
        new_job_id, error = comms.resume(ids[0], message)
        if error is not None:
            return _error(tool_call_id, "hub", error)
        return _text(
            tool_call_id,
            "hub",
            f"resumed {comms.label_of(ids[0])} as job {new_job_id}; it replays its own "
            "transcript before reading this instruction. Await it with 'wait'.",
            details={"op": "resume", "job_id": new_job_id, "resumed_from": ids[0]},
        )

    deliveries = []
    for job_id in ids:
        if params.op == "send":
            deliveries.append(comms.send(job_id, message))
        elif params.op == "steer":
            deliveries.append(comms.steer(job_id, message))
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
            "Talk to the subagents you launched with 'task': send a note, ask one a "
            "question and get its answer (use this to find out whether a quiet child is "
            "stuck), steer one onto a different course, cancel one, or resume a stopped "
            "one against its own transcript. Address them by job id, by label, or "
            '"all".'
        ),
        parameters=HubParams.model_json_schema(),
        # Write, like 'task' and 'wake': these ops redirect, kill and restart
        # autonomous work. The gate is per TOOL, not per op, so the tier is
        # the highest any op needs — 'resume' starts a child session, which is
        # exactly what 'task' asks the user to approve.
        approval_tier="write",
        # 'ask' blocks the turn on another agent's answer; running it beside
        # other tools would hold a shared slot for the whole timeout.
        concurrency="exclusive",
        interruptible=True,
        execute=execute_hub,
    )
