"""The `/copy` picker's target tree: assistant messages and their blocks.

A pure text transform. It takes transcript blocks and returns a tree of
:class:`CopyTarget` nodes — most recent assistant answer first, each drillable
into its fenced code blocks and `>`-quoted runs — and it imports nothing from
Textual or from ``widgets/``. That is not tidiness: it is what lets the whole
grammar be tested against hand-built messages with no running app, which is
the same reason the reference implementation keeps `utils/copy-targets.ts`
apart from `components/copy-selector.ts`.

Ported from oh-my-pi's `extractBlocks`/`buildCopyTargets`, with ONE deliberate
divergence, in the fence rule. See :func:`extract_blocks`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Protocol, Sequence, runtime_checkable

#: Cap on how many recent assistant messages the picker lists. Counted over
#: messages with non-empty text, so an answer that was only tool calls does
#: not consume budget.
MAX_MESSAGES = 50

#: Opening/closing fence marker. Deliberately the SAME pattern as
#: ``widgets/_copy_markdown._FENCE_RE`` — see :func:`extract_blocks` for why
#: this module re-states the grammar rather than importing that module's
#: ``classify``. The two are the only readers of this grammar; a change to one
#: is a change to both.
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
#: A blockquote line. One ``>`` level only, matching the reference: the marker
#: plus at most one following space comes off, so indentation INSIDE the quote
#: survives into the clipboard.
_QUOTE_RE = re.compile(r"^>(.*)$")
_WHITESPACE_RUN_RE = re.compile(r"\s+")


@runtime_checkable
class AssistantMessage(Protocol):
    """The minimal transcript surface needed to assemble copy targets.

    A ``Protocol`` rather than ``isinstance(block, AssistantBlock)`` for two
    reasons that pull the same way. It keeps this module free of any widget
    import, which is the claim the module's position outside ``widgets/``
    makes and which the import graph then enforces. And it is what makes the
    tree testable from plain objects — the reference interface (`CopySource`)
    exists for exactly that stated reason.

    ``is_truncated`` is what discriminates an assistant answer from the other
    blocks that also carry ``text()``: ``UserBlock``, ``NoticeBlock`` and
    ``PeerMessageBlock`` all have text and all inherit ``is_finalized``, and
    only an assistant answer can have ended early. Adding ``is_truncated`` to
    another block type would silently admit it here — which is what
    ``test_copy_targets.py`` pins.
    """

    def text(self) -> str: ...

    def is_finalized(self) -> bool: ...

    def is_truncated(self) -> bool: ...


@dataclass(frozen=True)
class MessageBlock:
    """One drillable block inside an assistant message."""

    kind: Literal["code", "quote"]
    #: De-prefixed for quotes; the fence body (markers excluded) for code.
    body: str
    #: Code only; ``""`` when the fence carried no info string.
    lang: str = ""


@dataclass(frozen=True)
class CopyTarget:
    """A node in the `/copy` picker tree.

    Frozen because the tree is built once, when the screen is pushed, and is
    never mutated: the picker SNAPSHOTS the transcript so a message settling
    mid-turn cannot shift rows under a cursor the user is already aiming with.
    New messages insert at the top, so a live rebuild would move every row.
    """

    #: Stable id (``msg:1``, ``msg:1:code:0``, ``msg:1:all``). Not used for
    #: cursor tracking today — the tree cannot change while the picker is open
    #: — but it is what a re-homing cursor would key on if it ever does.
    id: str
    label: str
    hint: str
    preview: str
    #: Text placed on the clipboard. ``None`` marks a node Enter refuses.
    #: Every node this module builds IS copyable; the field and its guard are
    #: kept as the seam the (unported) command targets would slot into, and
    #: they cost one ``if``.
    content: str | None
    #: The reference shows this per-target status after a copy. We do not: the
    #: clipboard receipt is shared with the drag and composer gestures so it
    #: cannot drift per gesture. Carried so re-enabling it is one line.
    copy_message: str
    #: Highlight lexer for the preview; ``None`` means plain text.
    language: str | None = None
    children: tuple["CopyTarget", ...] = ()
    #: Ours, not the reference's. The answer was cut off before it finished.
    truncated: bool = False


def extract_blocks(text: str) -> list[MessageBlock]:
    """Split assistant markdown into drillable blocks, in document order.

    Three properties, each of which a test pins:

    - **Fences mask their bodies.** A ``>`` line inside a code block is never
      a quote; the scan jumps past the whole block.
    - **An unclosed fence is ordinary text.** A streaming or aborted answer
      very often ends mid-fence, and that half block must not become a code
      target whose body is the rest of the message.
    - **The closing marker must match the opening character.** This is the
      divergence from the reference, whose ``/^```/`` closes a ``~~~`` block
      on a backtick run and does not recognise ``~~~`` as an opener at all.
      ``_copy_markdown.classify`` already got this right for the drag-select
      path; porting the laxer rule would import a defect into a repo that
      does not have it.
    """
    blocks: list[MessageBlock] = []
    lines = text.split("\n")
    quote: list[str] = []

    def flush_quote() -> None:
        if quote:
            blocks.append(MessageBlock(kind="quote", body="\n".join(quote)))
            quote.clear()

    index = 0
    while index < len(lines):
        line = lines[index]
        opened = _FENCE_RE.match(line)
        if opened is not None:
            close = _find_close_fence(lines, index, opened.group(1)[0])
            if close is not None:
                flush_quote()
                blocks.append(
                    MessageBlock(
                        kind="code",
                        body="\n".join(lines[index + 1 : close]),
                        lang=line[opened.end() :].strip(),
                    )
                )
                index = close + 1
                continue
            # No closer: fall through and let the line be treated as text.

        quoted = _QUOTE_RE.match(line)
        if quoted is not None:
            body = quoted.group(1)
            quote.append(body[1:] if body.startswith(" ") else body)
        else:
            flush_quote()
        index += 1

    flush_quote()
    return blocks


def _find_close_fence(lines: list[str], start: int, fence_char: str) -> int | None:
    """Index of the line closing the fence opened at ``start``, or ``None``.

    A closer is a marker run of the SAME character with nothing else on the
    line — ``classify``'s rule. An info string after the marker makes it
    another opener, not a closer.
    """
    for index in range(start + 1, len(lines)):
        line = lines[index]
        marker = _FENCE_RE.match(line)
        if marker is None:
            continue
        stripped = line.strip()
        if marker.group(1)[0] == fence_char and set(stripped) <= {fence_char}:
            return index
    return None


def plural_lines(text: str) -> str:
    """``"1 line"`` / ``"2 lines"``; empty text is ``"0 lines"``."""
    count = 0 if not text else len(text.split("\n"))
    return f"{count} line" if count == 1 else f"{count} lines"


def first_line(text: str) -> str:
    """First non-empty line, whitespace-collapsed — a message's label.

    The all-blank fallback is unreachable through :func:`build_copy_targets`,
    which skips blank messages before it gets here. Kept rather than raising,
    because a label is not worth an exception.
    """
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped:
            return _WHITESPACE_RUN_RE.sub(" ", stripped)
    return _WHITESPACE_RUN_RE.sub(" ", text.strip())


def _block_hint(block: MessageBlock) -> str:
    lines = plural_lines(block.body)
    return f"{block.lang} · {lines}" if block.lang else lines


def _message_hint(text: str, code_count: int, quote_count: int, truncated: bool) -> str:
    """``"truncated · 12 lines · 1 code"`` — absent kinds omitted.

    The line count is of the WHOLE message, not of its blocks.

    ``truncated`` leads rather than trails. The hint is right-aligned and it
    is the LABEL that gets truncated to fit beside it, so a trailing marker is
    the first thing a narrow terminal cuts — exactly when a user most needs to
    know the answer was cut off. Leading also groups it with the other
    structural facts instead of reading like a third block count.
    """
    parts = ["truncated"] if truncated else []
    parts.append(plural_lines(text))
    if code_count > 0:
        parts.append(f"{code_count} code")
    if quote_count > 0:
        parts.append(f"{quote_count} quote")
    return " · ".join(parts)


def _message_target(text: str, rank: int, truncated: bool) -> CopyTarget:
    """One assistant message: a leaf when it has no blocks, else a group.

    The group node itself copies the whole message, so drilling in is never
    the only way to get the answer out.
    """
    node_id = f"msg:{rank}"
    label = first_line(text)
    blocks = extract_blocks(text)
    # Rank 1 is the message a bare `/copy` used to take, and saying so is what
    # tells the user the picker did not change that answer.
    message_copy = (
        "Copied last message to clipboard" if rank == 1 else "Copied message to clipboard"
    )

    code = [block for block in blocks if block.kind == "code"]
    quotes = [block for block in blocks if block.kind == "quote"]
    children: list[CopyTarget] = []

    # Children are NOT marked truncated even when their message is. A closed
    # fence inside a cut-off answer is complete — the truncation ended the
    # message, not that block — so marking it would be a false claim about
    # what is on the clipboard. A genuinely half-written trailing fence never
    # becomes a block at all (see `extract_blocks`).
    code_index = 0
    quote_index = 0
    for block in blocks:
        if block.kind == "code":
            children.append(
                CopyTarget(
                    id=f"{node_id}:code:{code_index}",
                    label=f"Block {code_index + 1}",
                    hint=_block_hint(block),
                    preview=block.body,
                    content=block.body,
                    copy_message=f"Copied code block {code_index + 1} to clipboard",
                    language=block.lang or None,
                )
            )
            code_index += 1
        else:
            children.append(
                CopyTarget(
                    id=f"{node_id}:quote:{quote_index}",
                    label=f"Quote {quote_index + 1}",
                    hint=plural_lines(block.body),
                    preview=block.body,
                    content=block.body,
                    copy_message=f"Copied quote block {quote_index + 1} to clipboard",
                )
            )
            quote_index += 1

    # "All N" only when there is more than one of a kind: with a single block
    # it would be a second row copying byte-for-byte what the first one does.
    if len(code) > 1:
        combined = "\n\n".join(block.body for block in code)
        children.append(
            CopyTarget(
                id=f"{node_id}:all",
                label=f"All {len(code)} blocks",
                hint=plural_lines(combined),
                preview=combined,
                content=combined,
                copy_message=f"Copied {len(code)} code blocks to clipboard",
            )
        )
    if len(quotes) > 1:
        combined = "\n\n".join(block.body for block in quotes)
        children.append(
            CopyTarget(
                id=f"{node_id}:all-quotes",
                label=f"All {len(quotes)} quotes",
                hint=plural_lines(combined),
                preview=combined,
                content=combined,
                copy_message=f"Copied {len(quotes)} quote blocks to clipboard",
            )
        )

    return CopyTarget(
        id=node_id,
        label=label,
        hint=_message_hint(text, len(code), len(quotes), truncated),
        preview=text,
        content=text,
        copy_message=message_copy,
        children=tuple(children),
        truncated=truncated,
    )


def build_copy_targets(blocks: Sequence[object]) -> list[CopyTarget]:
    """The `/copy` tree over ``blocks``, most recent assistant answer first.

    ``Sequence[object]`` is deliberate, not laziness. The narrower
    ``Sequence[AssistantMessage]`` would REJECT the only real caller: the
    transcript hands over a ``list[TranscriptBlock]`` of mixed user, notice,
    tool and assistant blocks, and pyright rejects it because
    ``TranscriptBlock`` has neither ``text`` nor ``is_truncated`` (verified,
    not assumed). Selecting the assistant answers out of that mixture IS this
    function's job, so the parameter has to admit the mixture; the
    ``isinstance`` against the Protocol below is where the type is actually
    established. Typing it to ``TranscriptBlock`` instead would import a
    widget and break the layering this module's position asserts.

    ``blocks`` is the transcript's own block list, walked in REVERSE: append
    order is the order the reader sees, and a resumed conversation replays its
    history into the same column, so "most recent" means last-appended rather
    than last-sent.

    Three filters, mirroring the last-message walk this command replaces:
    the block must be an assistant answer (:class:`AssistantMessage`), it must
    be finalized, and its text must be non-blank. ``is_finalized`` means
    IMMUTABLE, not COMPLETE — an aborted answer is frozen exactly as a clean
    one is — so a truncated message IS listed, and marked. Skipping it would
    hide the message the user was reading and most likely meant.

    The caller passes ``_transcript_view().blocks()``, never a type query: the
    full-page subagent view mounts a SECOND ``TranscriptView`` for the child's
    conversation, and a query would start raising exactly while it is open.
    """
    targets: list[CopyTarget] = []
    rank = 0
    for block in reversed(list(blocks)):
        if rank >= MAX_MESSAGES:
            break
        if not isinstance(block, AssistantMessage):
            continue
        if not block.is_finalized():
            continue
        text = block.text()
        if not text.strip():
            continue
        rank += 1
        targets.append(_message_target(text, rank, block.is_truncated()))
    return targets


@dataclass(frozen=True)
class FlatNode:
    """One drawn row: a target plus the tree geometry that indents it."""

    target: CopyTarget
    depth: int
    #: Last among its siblings — drives ``└─`` against ``├─``.
    is_last: bool
    #: Per-ancestor: does the ancestor at that level have a following sibling?
    #: A ``│`` guide is drawn in its gutter cell when it does.
    ancestor_has_next: tuple[bool, ...] = field(default=())


def flatten_targets(roots: Sequence[CopyTarget]) -> list[FlatNode]:
    """Depth-first flatten of ``roots`` into the rows the picker draws.

    Lives here rather than on the screen because it is the tree's shape, not
    the widget's: the picker's row count, page step and cursor bounds are all
    derived from it, and pinning it without a running app is worth more than
    keeping it beside the paint code.
    """
    out: list[FlatNode] = []

    def walk(nodes: Sequence[CopyTarget], depth: int, ancestor_has_next: tuple[bool, ...]) -> None:
        for index, target in enumerate(nodes):
            is_last = index == len(nodes) - 1
            out.append(
                FlatNode(
                    target=target,
                    depth=depth,
                    is_last=is_last,
                    ancestor_has_next=ancestor_has_next,
                )
            )
            if target.children:
                walk(target.children, depth + 1, ancestor_has_next + (not is_last,))

    walk(roots, 0, ())
    return out
