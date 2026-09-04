"""Public compaction API consumed by the session.

Everything the session stream imports from ``local_operator.compaction.api``
lives here (settings, threshold math, cut point, summary prompt, pruning,
token estimation). The submodules split the implementation; this module is
the stable surface.

Summary-prompt contract: the instruction body is read from
``local_operator/prompts_md/compaction_summary.md`` (owned by the tools
stream) as a plain file via ``importlib.resources``. The template uses
``{{transcript}}``, ``{{#if previous_summary}}…{{previous_summary}}…{{/if}}``
and an optional ``{{#if files}}`` appendix; we substitute those inline with a
minimal renderer so this module has zero dependency on ``prompts_api`` and
stays importable before that module lands. A missing template raises
:class:`FileNotFoundError` — the template is part of the contract, and a
silent lossy fallback would summarize against the wrong instructions.
"""

from __future__ import annotations

import functools
import importlib.resources
import json
import re
from typing import Any, Awaitable, Callable, Sequence

from pydantic import BaseModel

from local_operator.harness.types import AgentMessage, CustomMessage, Message, ToolCall

from .advisor import (
    ADVISOR_MAX_REASON_CHARS,
    ADVISOR_SYSTEM_PROMPT,
    CompactionHint,
    build_advisor_prompt,
    parse_hint,
    validate_hint,
)
from .cutpoint import (
    PRESERVED_USER_TURN_KEY,
    extract_preserved_user_turns,
    find_cut_point,
    prepare_partitions,
    task_boundary_floor,
)
from .pruning import (
    MIN_PRUNE_TOKENS,
    SUPERSEDED_NOTICE,
    USELESS_NOTICE,
    _is_useless,
    compute_suffix_tokens,
    prune_tool_outputs,
    shed_frames_to_wire_budget,
)
from .thresholds import (
    DEFAULT_WIRE_BYTES_BUDGET,
    DEFAULT_WIRE_BYTES_TRIGGER,
    RECOVERY_BAND,
    WIRE_RECOVERY_BAND,
    CompactionSettings,
    cleared_headroom,
    cleared_wire_headroom,
    compaction_context_tokens,
    effective_reserve_tokens,
    resolve_advisor_floor_tokens,
    resolve_strategy,
    resolve_threshold_tokens,
    resolve_wire_bytes_budget,
    resolve_wire_bytes_trigger,
    should_compact,
)
from .tokens import (
    IMAGE_TOKEN_ESTIMATE,
    OFFLOAD_MIN_CHARS,
    _encode_len,
    clear_estimate_cache,
    estimate_messages_tokens,
    estimate_tokens,
    estimate_wire_bytes,
    history_chars,
    invalidate_message_cache,
    messages_tokens_upper_bound,
    register_invalidator,
    truncate_to_tokens,
)

__all__ = [
    "ADVISOR_MAX_REASON_CHARS",
    "ADVISOR_SYSTEM_PROMPT",
    "CompactionHint",
    "CompactionResult",
    "CompactionSettings",
    "IMAGE_TOKEN_ESTIMATE",
    "MAX_SUMMARY_TOKENS",
    "MIN_PRUNE_TOKENS",
    "PRESERVED_USER_TURN_KEY",
    "RECOVERY_BAND",
    "WIRE_RECOVERY_BAND",
    "DEFAULT_WIRE_BYTES_BUDGET",
    "DEFAULT_WIRE_BYTES_TRIGGER",
    "SUMMARIZATION_SYSTEM_PROMPT",
    "SUPERSEDED_NOTICE",
    "TOOL_ARGS_MAX_CHARS",
    "TOOL_RESULT_MAX_CHARS",
    "USELESS_NOTICE",
    "build_advisor_prompt",
    "build_compaction_prompt",
    "clear_estimate_cache",
    "cleared_headroom",
    "cleared_wire_headroom",
    "compaction_context_tokens",
    "compute_suffix_tokens",
    "effective_reserve_tokens",
    "estimate_messages_tokens",
    "estimate_tokens",
    "estimate_wire_bytes",
    "extract_file_ops_from_messages",
    "extract_preserved_user_turns",
    "find_cut_point",
    "history_chars",
    "OFFLOAD_MIN_CHARS",
    "format_file_operations",
    "invalidate_message_cache",
    "messages_tokens_upper_bound",
    "parse_hint",
    "prepare_partitions",
    "prune_tool_outputs",
    "shed_frames_to_wire_budget",
    "register_invalidator",
    "render_file_operations",
    "resolve_advisor_floor_tokens",
    "resolve_strategy",
    "resolve_threshold_tokens",
    "resolve_wire_bytes_budget",
    "resolve_wire_bytes_trigger",
    "serialize_conversation",
    "should_compact",
    "summarize_messages",
    "task_boundary_floor",
    "upsert_file_operations",
    "validate_hint",
]

#: Tool results in the serialized transcript are truncated to this many
#: characters (``TOOL_RESULT_MAX_CHARS``): the summary model does not
#: need full outputs, and unbounded results are the main context bloat.
TOOL_RESULT_MAX_CHARS = 2000

#: Serialized tool-call arguments are truncated to this many characters.
TOOL_ARGS_MAX_CHARS = 500

#: Cap on the archived-history edges a snapcompact marker serializes into a
#: text transcript (``_serialize_message``). Generous relative to the other
#: slots because the edges ARE the content the marker stands for, but bounded
#: because they can reach ~71k chars each and a direct consumer of
#: ``serialize_conversation`` must be able to budget its prompt. Tail-biased
#: on truncation: chaining wants the newest history most.
ARCHIVE_EDGES_MAX_CHARS = 8000


#: Hard cap on a generated summary (``MAX_SUMMARY_TOKENS``). The
#: complete_fn has no max-tokens knob yet, so the cap is enforced post-hoc:
#: summaries above it are truncated by tokens with a
#: ``[summary truncated]`` marker (see :func:`summarize_messages`).
MAX_SUMMARY_TOKENS = 16384

_IF_BLOCK_RE = re.compile(r"\{\{#if (\w+)\}\}(.*?)\{\{/if\}\}", re.DOTALL)
_VAR_RE = re.compile(r"\{\{(\w+)\}\}")

#: File-operation summary cap: at most this many files are rendered before
#: ``[…N files elided…]`` (``FILE_OPERATION_SUMMARY_LIMIT``).
_FILE_OPERATION_SUMMARY_LIMIT = 20

#: Read-tool selector grammar, mirrored from ``splitReadSelector`` /
#: ``stripReadSelector`` (utils.ts). A trailing ``:chunk`` is a selector only
#: if it is a line-range list (``50``, ``50-200``, ``50+10``, ``5-16,960-973``,
#: ``..`` alias), ``raw``, or ``conflicts`` — alone or as a ``range:raw`` /
#: ``raw:range`` compound.
_RANGE_CHUNK_SRC = r"L?\d+(?:(?:[-+]|\.\.)L?\d+|-|\.\.)?"
_RANGE_LIST_SRC = rf"{_RANGE_CHUNK_SRC}(?:,{_RANGE_CHUNK_SRC})*"
_READ_SELECTOR_RE = re.compile(rf"^(?:{_RANGE_LIST_SRC}|raw|conflicts)$", re.IGNORECASE)
_READ_RANGE_ONLY_RE = re.compile(rf"^{_RANGE_LIST_SRC}$", re.IGNORECASE)
_READ_RAW_ONLY_RE = re.compile(r"^raw$", re.IGNORECASE)

#: ``scheme://`` paths (internal URIs and web URLs) are session-scoped or
#: remote resources, not re-groundable files — they stay out of the ``<files>``
#: summary (``isUrlSchemePath``).
_URL_SCHEME_RE = re.compile(r"[a-z][a-z0-9+.-]*://", re.IGNORECASE)

#: Legacy file-operation tags are stripped from prior summaries so a summary
#: written before the combined ``<files>`` tag self-heals on the next
#: compaction (``stripFileOperationTags``).
_FILE_TAG_RE = re.compile(r"<files>.*?</files>\s*", re.DOTALL)
_READ_FILES_TAG_RE = re.compile(r"<read-files>.*?</read-files>\s*", re.DOTALL)
_MODIFIED_FILES_TAG_RE = re.compile(r"<modified-files>.*?</modified-files>\s*", re.DOTALL)

#: Truncation marker appended when a summary exceeds MAX_SUMMARY_TOKENS.
SUMMARY_TRUNCATED_MARKER = "[summary truncated]"

#: System prompt for the summarization call (``summarization-system.md``).
SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context compaction summarizer. Summarize conversations between "
    "users and AI coding assistants. Produce structured summaries in the exact "
    "specified format.\n"
    "\n"
    "NEVER continue the conversation. NEVER respond to questions in it. Output "
    "ONLY the structured summary."
)


class CompactionResult(BaseModel):
    """Outcome of one compaction pass, recorded by the session.

    ``first_kept_index`` points into the message list that was compacted:
    replay is the summary message plus messages from that index onward.
    ``preserve_data`` is the strategy-specific replay payload (e.g.
    ``{"snapcompact": <archive dump>}``) stored on the compaction entry so a
    later compaction can rebuild from it instead of re-summarizing.
    """

    summary: str
    first_kept_index: int
    tokens_before: int
    preserve_data: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _render_tool_call(call: ToolCall) -> str:
    """``name(args)`` with arguments truncated — call scaffolding only."""
    args = call.raw_arguments
    if args is None:
        try:
            args = json.dumps(call.arguments, sort_keys=True)
        except (TypeError, ValueError):
            args = str(call.arguments)
    if len(args) > TOOL_ARGS_MAX_CHARS:
        args = args[:TOOL_ARGS_MAX_CHARS] + f"[... {len(args) - TOOL_ARGS_MAX_CHARS} more chars]"
    return f"{call.name}({args})"


def _useless_tool_call_ids(messages: Sequence[AgentMessage]) -> set[str]:
    """Call ids answered by useless-flagged tool results.

    Useless results AND their paired calls are dropped from the summary
    input entirely (``serializeConversation``): both carry no signal.
    """
    ids: set[str] = set()
    for message in messages:
        if not isinstance(message, Message) or message.role != "tool":
            continue
        if _is_useless(message) and message.tool_call_id:
            ids.add(message.tool_call_id)
    return ids


def _serialize_message(message: AgentMessage, drop_call_ids: set[str]) -> str | None:
    """One transcript block, or None to drop the message entirely."""
    if isinstance(message, CustomMessage):
        if message.custom_type == "compaction_summary":
            # A snapcompact marker's summary is reading instructions for its
            # image frames, not a digest — chaining IT into a later text
            # summarization would replace the archived history with constant
            # boilerplate. The archive's text edges are the real transcript
            # (newest and oldest slices), so serialize those instead. The
            # session's own summarize path never reaches this branch (it
            # converts markers to rendered content first); this guards direct
            # consumers of serialize_conversation handed raw transcript
            # vocabulary.
            preserve = message.details.get("preserve_data") or {}
            snap = preserve.get("snapcompact") if isinstance(preserve, dict) else None
            if isinstance(snap, dict):
                edges = [
                    edge
                    for edge in (snap.get("text_head"), snap.get("text_tail"))
                    if isinstance(edge, str) and edge.strip()
                ]
                if edges:
                    joined = "\n[...]\n".join(edges)
                    # Bounded like every other free-text slot here, but far
                    # more generously (the edges are up to ~71k chars EACH
                    # under the Gemini shape, and this serializer's other
                    # slots cap at 2,000): a direct consumer building a
                    # prompt from a compacted conversation must not inherit
                    # an unbounded ~36k-token block. Tail-biased, because
                    # chaining wants the NEWEST history most.
                    if len(joined) > ARCHIVE_EDGES_MAX_CHARS:
                        kept = joined[-ARCHIVE_EDGES_MAX_CHARS:]
                        dropped = len(joined) - ARCHIVE_EDGES_MAX_CHARS
                        joined = f"[... {dropped} older characters truncated]\n{kept}"
                    return f"[Previously archived history (edges)]\n{joined}"
            summary = message.details.get("summary", "")
            return f"[Previous compaction summary]\n{summary}"
        return f"[{message.custom_type}]"

    parts: list[str] = []
    if message.role == "user":
        parts.append(f"[User]\n{message.text}")
    elif message.role == "assistant":
        text = message.text
        calls = [c for c in message.tool_calls if c.id not in drop_call_ids]
        if text:
            parts.append(f"[Assistant]\n{text}")
        if calls:
            rendered = "; ".join(_render_tool_call(c) for c in calls)
            parts.append(f"[Assistant tool calls] {rendered}")
        if not parts:
            return None
    else:  # tool result
        if message.tool_call_id and message.tool_call_id in drop_call_ids:
            return None
        text = message.text
        if len(text) > TOOL_RESULT_MAX_CHARS:
            kept = text[:TOOL_RESULT_MAX_CHARS]
            text = f"{kept}\n[... {len(text) - TOOL_RESULT_MAX_CHARS} more characters truncated]"
        name = message.tool_name or "tool"
        prefix = f"[Tool result: {name}]"
        if message.is_error:
            prefix = f"[Tool ERROR: {name}]"
        parts.append(f"{prefix}\n{text}")
    return "\n".join(parts)


def serialize_conversation(messages: Sequence[AgentMessage]) -> str:
    """Deterministic text transcript of ``messages`` for the summarizer.

    Tool results truncate at :data:`TOOL_RESULT_MAX_CHARS` with an explicit
    ``[... N more characters truncated]`` tail; useless-flagged results and
    their paired calls are dropped entirely.
    """
    drop_ids = _useless_tool_call_ids(messages)
    blocks: list[str] = []
    for message in messages:
        block = _serialize_message(message, drop_ids)
        if block:
            blocks.append(block)
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# File operations (utils.ts extractFileOpsFromMessage / upsertFileOperations)
# ---------------------------------------------------------------------------


def _split_read_selector(path: str) -> tuple[str, str | None]:
    """``(base, selector)`` — the read tool's selector grammar (see the
    ``_READ_SELECTOR_RE`` comment). A trailing ``:chunk`` counts only if it
    is a line-range list, ``raw``, or ``conflicts``, alone or compounded."""
    colon = path.rfind(":")
    if colon <= 0:
        return path, None
    candidate = path[colon + 1 :]
    if not _READ_SELECTOR_RE.match(candidate):
        return path, None
    base, selector = path[:colon], candidate
    inner = base.rfind(":")
    if inner > 0:
        inner_candidate = base[inner + 1 :]
        inner_is_raw = bool(_READ_RAW_ONLY_RE.match(inner_candidate))
        outer_is_raw = bool(_READ_RAW_ONLY_RE.match(candidate))
        inner_is_range = bool(_READ_RANGE_ONLY_RE.match(inner_candidate))
        outer_is_range = bool(_READ_RANGE_ONLY_RE.match(candidate))
        if (inner_is_raw and outer_is_range) or (inner_is_range and outer_is_raw):
            selector = f"{inner_candidate}:{candidate}"
            base = base[:inner]
    return base, selector


def _strip_read_selector(path: str) -> str:
    """Strip a trailing read selector so the same file read with different
    line ranges dedupes to one ``<files>`` entry."""
    return _split_read_selector(path)[0]


def _is_url_scheme_path(path: str) -> bool:
    """``scheme://`` targets are not re-groundable files (see ``_URL_SCHEME_RE``)."""
    return bool(_URL_SCHEME_RE.search(path))


def extract_file_ops_from_messages(messages: Sequence[AgentMessage]) -> dict[str, set[str]]:
    """File-operation sets scanned from assistant tool calls.

    Returns ``{"read": ..., "written": ..., "edited": ...}`` sets keyed on the
    tool-call ``arguments['path']``: ``read`` calls land in the read set (line
    selectors stripped), ``write``/``edit`` in written/edited. ``scheme://``
    paths are dropped — session-scoped or remote resources do not belong in
    the ``<files>`` summary.
    """
    ops: dict[str, set[str]] = {"read": set(), "written": set(), "edited": set()}
    for message in messages:
        if not isinstance(message, Message) or message.role != "assistant":
            continue
        for call in message.tool_calls:
            arguments = call.arguments if isinstance(call.arguments, dict) else {}
            path = arguments.get("path")
            if not isinstance(path, str) or not path:
                continue
            if _is_url_scheme_path(path):
                continue
            if call.name == "read":
                ops["read"].add(_strip_read_selector(path))
            elif call.name == "write":
                ops["written"].add(path)
            elif call.name == "edit":
                ops["edited"].add(path)
    return ops


def _format_grouped_paths(paths: Sequence[str], mode: dict[str, str]) -> str:
    """Prefix-folded directory tree (``formatGroupedPaths``): single-child
    directory chains fold into one ``#`` header per level, files list bare
    under the deepest directory that owns them, annotated `` (Read)`` /
    `` (Write)`` / `` (RW)``."""

    class _Node:
        __slots__ = ("files", "subdirs")

        def __init__(self) -> None:
            # (display name, original full path) — annotation looks up the
            # ORIGINAL path, never a reconstructed one (absolute paths carry a
            # leading slash the segments drop).
            self.files: list[tuple[str, str]] = []
            self.subdirs: dict[str, "_Node"] = {}

    root = _Node()
    for path in paths:
        normalized = path.replace("\\", "/")
        if _is_url_scheme_path(normalized):
            root.files.append((normalized, path))
            continue
        segments = [s for s in normalized.split("/") if s]
        node = root
        for segment in segments[:-1]:
            node = node.subdirs.setdefault(segment, _Node())
        if segments:
            node.files.append((segments[-1], path))

    lines: list[str] = []

    def walk(node: _Node, depth: int) -> None:
        for name, original in node.files:
            suffix = f" ({mode[original]})" if original in mode else ""
            lines.append(f"{name}{suffix}")
        for name, child in node.subdirs.items():
            parts = [name]
            cursor = child
            # Fold single-child chains with no own files into one header.
            while not cursor.files and len(cursor.subdirs) == 1:
                only = next(iter(cursor.subdirs))
                parts.append(only)
                cursor = cursor.subdirs[only]
            lines.append(f"{'#' * (depth + 1)} {'/'.join(parts)}/")
            walk(cursor, depth + 1)

    walk(root, 0)
    return "\n".join(lines)


def format_file_operations(
    read_files: Sequence[str],
    modified_files: Sequence[str],
    read_set: set[str] | None = None,
) -> str:
    """Render file operations as a folded tree with ``(Read)``/``(Write)``/
    ``(RW)`` markers, capped at 20 files with ``[…N files elided…]``. Empty
    when there is nothing to render.
    """
    if not read_files and not modified_files:
        return ""
    mode: dict[str, str] = {file: "Read" for file in read_files}
    for file in modified_files:
        mode[file] = "RW" if read_set is not None and file in read_set else "Write"
    all_files = sorted(mode)
    shown = all_files[:_FILE_OPERATION_SUMMARY_LIMIT]
    rendered = _format_grouped_paths(shown, mode)
    if len(all_files) > _FILE_OPERATION_SUMMARY_LIMIT:
        rendered += f"\n[…{len(all_files) - _FILE_OPERATION_SUMMARY_LIMIT} files elided…]"
    return rendered


def _compute_file_lists(ops: dict[str, set[str]]) -> tuple[list[str], list[str]]:
    """Final ``(read_only, modified)`` lists from file-operation sets.

    ``modified`` is edited ∪ written, URL-scheme paths filtered;
    read-only is the read set minus modified, sorted."""
    modified = {f for f in (*ops["edited"], *ops["written"]) if not _is_url_scheme_path(f)}
    read_only = sorted(f for f in ops["read"] if not _is_url_scheme_path(f) and f not in modified)
    return read_only, sorted(modified)


def render_file_operations(messages: Sequence[AgentMessage]) -> str:
    """Rendered ``{{files}}`` payload for a message list: extract file ops
    from assistant tool calls, then fold them into the marker tree. Empty
    string when the transcript touched no files."""
    ops = extract_file_ops_from_messages(messages)
    read_only, modified = _compute_file_lists(ops)
    return format_file_operations(read_only, modified, ops["read"])


def upsert_file_operations(
    summary: str,
    read_files: Sequence[str],
    modified_files: Sequence[str],
    read_set: set[str] | None = None,
) -> str:
    """Append the ``<files>`` appendix to ``summary``.

    Legacy ``<files>``/``<read-files>``/``<modified-files>`` tags are
    stripped first so old summaries self-heal."""
    base = _FILE_TAG_RE.sub("", summary)
    base = _READ_FILES_TAG_RE.sub("", base)
    base = _MODIFIED_FILES_TAG_RE.sub("", base).rstrip()
    appendix = format_file_operations(read_files, modified_files, read_set)
    if not appendix:
        return base
    if not base:
        return f"<files>\n{appendix}\n</files>"
    return f"{base}\n\n<files>\n{appendix}\n</files>"


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def _render_template(template: str, data: dict[str, str]) -> str:
    """Minimal ``{{var}}`` / ``{{#if var}}…{{/if}}`` substitution.

    Mirrors the subset of ``prompts_api.render_template`` the compaction
    template uses, implemented inline so compaction never imports the tools
    stream. Unknown variables and falsey ``#if`` blocks collapse to empty.
    """

    def _if_sub(match: re.Match[str]) -> str:
        name, body = match.group(1), match.group(2)
        return body if data.get(name) else ""

    rendered = _IF_BLOCK_RE.sub(_if_sub, template)
    return _VAR_RE.sub(lambda m: data.get(m.group(1), ""), rendered)


@functools.lru_cache(maxsize=1)
def _load_instruction_template() -> str:
    """Plain file read of prompts_md/compaction_summary.md, cached (RC-24).

    ``importlib.resources`` (not ``__file__`` joins) so the template resolves
    from the installed package, not the caller's cwd. A missing or empty
    template raises :class:`FileNotFoundError`: the template is part of the
    prompt contract, and a silent fallback would summarize against the wrong
    instructions.
    """
    try:
        resource = importlib.resources.files("local_operator").joinpath(
            "prompts_md/compaction_summary.md"
        )
        text = resource.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, OSError, TypeError) as exc:
        raise FileNotFoundError(
            "prompts_md/compaction_summary.md is missing: the compaction summary "
            "template is part of the prompt contract and cannot be substituted."
        ) from exc
    if not text.strip():
        raise FileNotFoundError(
            "prompts_md/compaction_summary.md is empty: the compaction summary "
            "template is part of the prompt contract and cannot be substituted."
        )
    return text


def build_compaction_prompt(
    messages_to_summarize: Sequence[AgentMessage],
    previous_summary: str | None = None,
    files: str | None = None,
) -> str:
    """Full summarization prompt for one compaction pass.

    The serialized conversation is wrapped in ``<conversation>``; a prior
    summary is wrapped in ``<previous-summary>``. The instruction body comes
    from ``prompts_md/compaction_summary.md`` (see module docstring) rendered
    with the transcript; iterative compactions additionally pass the previous
    summary so the template can fold it in.

    ``files`` is the rendered file-operation tree substituted into the
    template's ``{{files}}`` slot. ``None`` (the default) auto-extracts file
    ops from ``messages_to_summarize`` (:func:`render_file_operations`) so the
    live template's ``{{#if files}}`` block fires; pass ``""`` to suppress.
    """
    transcript = serialize_conversation(messages_to_summarize)
    conversation_block = f"<conversation>\n{transcript}\n</conversation>"

    template = _load_instruction_template()
    files_block = render_file_operations(messages_to_summarize) if files is None else files
    data: dict[str, str] = {"transcript": conversation_block, "files": files_block}
    if previous_summary:
        data["previous_summary"] = f"<previous-summary>\n{previous_summary}\n</previous-summary>"
    else:
        data["previous_summary"] = ""
    return _render_template(template, data).strip()


async def summarize_messages(
    messages_to_summarize: Sequence[AgentMessage],
    complete_fn: Callable[[str, str], Awaitable[str]],
    previous_summary: str | None = None,
    files: str | None = None,
) -> str:
    """Run one summarization call and return the summary text.

    ``complete_fn(system, prompt) -> summary`` is provided by the session
    (it owns model selection/credentials). The system prompt is
    :data:`SUMMARIZATION_SYSTEM_PROMPT`; the prompt is
    :func:`build_compaction_prompt` (``files`` passes through to its
    ``{{files}}`` slot; ``None`` auto-extracts from the messages).

    The result is capped at :data:`MAX_SUMMARY_TOKENS` tokens: ``complete_fn``
    has no max-tokens knob yet, so the cap is enforced post-hoc by truncating
    the output and appending :data:`SUMMARY_TRUNCATED_MARKER`.
    """
    prompt = build_compaction_prompt(messages_to_summarize, previous_summary, files)
    summary = (await complete_fn(SUMMARIZATION_SYSTEM_PROMPT, prompt)).strip()
    return _cap_summary_tokens(summary)


def _cap_summary_tokens(summary: str) -> str:
    """Post-hoc :data:`MAX_SUMMARY_TOKENS` enforcement (RC-18) mirroring the
    maxTokens knob, which ``complete_fn`` does not expose yet."""
    if _encode_len(summary) <= MAX_SUMMARY_TOKENS:
        return summary
    truncated = truncate_to_tokens(summary, MAX_SUMMARY_TOKENS).rstrip()
    return f"{truncated}\n{SUMMARY_TRUNCATED_MARKER}"
