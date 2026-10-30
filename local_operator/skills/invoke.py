"""Manual skill invocation: the ``$name`` trigger and the payload block.

Semantic routing (:mod:`local_operator.skills.index`) decides which skills a
turn *probably* needs. This module is the other half: the user saying which
skills they want, by name, with no embedder in the loop.

``$research fix the login bug`` loads the ``research`` SKILL.md body and hands
it to the model together with the request. Three properties are deliberate:

- **Deterministic.** The body is injected as context, not suggested as a
  ``read skill://`` the model may skip. "Manual invocation" that the model can
  decline is not invocation, it is a hint.
- **Hidden skills are reachable.** A skill with ``hide: true`` /
  ``disable-model-invocation`` is excluded from semantic routing on purpose,
  and naming it explicitly is exactly the "never auto-select, let me fire it"
  case that flag describes, so :func:`plan_attachments` does not filter on
  ``hide``.
- **The frozen block is untouched.** Selection is frozen per session for
  prompt-cache warmth and re-opens only on compaction
  (``session_factory._render_knowledge_block``). Injecting at the MESSAGE
  level rides the volatile tail that changes every turn anyway, so a manual
  invocation costs nothing in cache warmth on the turns around it.

WHERE THE TRIGGER LIVES
-----------------------
Not here. :func:`local_operator.sigils.skill_mentions` owns "does this token
fire", because three surfaces ask it (this resolver, the composer's ink and the
picker's evidence gate) and two of them must not import this module — see the
import note at the block grammar in ``sigils.py``. The rule there is a boundary
``$name`` anywhere in the message, resolved against the discovered vocabulary;
this module only decides what a resolved mention is turned into.

THE PAYLOAD
-----------
The user's text goes out VERBATIM and the bodies ride a tail block:

    <user text>
    <BLANK LINE>
    <operator-skills>
    <BLANK LINE>
    <preamble>
    <BLANK LINE>
    <skill name="…">body</skill>
    ...
    </operator-skills>

A tail block on the same ``TextContent`` rather than a new content block type:
every consumer of a user row already takes a string and calls
``harness.rows.user_row_text``, which strips complete blocks. The alternative
(prompt-cache, provider serializers, mobile projection, the TypeScript client)
is recorded in ``docs/design/inline-skill-invocation.md``, D1.

REPLAY
------
The wire grammar above is FROZEN once persisted: changing the preamble orphans
old transcripts. :func:`local_operator.sigils.skill_block_spans` is the reader,
and it requires the marker, the preamble line and the closer — a marker the
operator merely quoted is not a block and is never hidden. The older
``<skill … invocation="…">`` payload stays readable forever via
:func:`typed_line_of`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple, TypeGuard

from local_operator.sigils import (
    SKILL_BLOCK_CLOSE,
    SKILL_BLOCK_JOIN,
    SKILL_BLOCK_OPEN,
    SKILL_BLOCK_PREAMBLE,
    SkillMention,
    defuse_skill_block,
    skill_block_names,
    skill_block_spans,
    skill_mentions,
)
from local_operator.skills.discovery import Skill

__all__ = [
    "SKILL_BODY_BUDGET_BYTES",
    "ExpandedInvocation",
    "SkillAttachment",
    "SkillPayloadTooLarge",
    "SkippedBody",
    "expand_invocation",
    "plan_attachments",
    "render_skill_block",
    "skill_block_names",
    "skill_block_spans",
    "skill_body_has_content",
    "typed_line_of",
]

#: The most a single message may attach as skill bodies, in WIRE bytes.
#:
#: A BOUND, not a measurement: a relay control frame is one JSON line with a
#: 1 MiB ceiling (``session.runtime.server._MAX_LINE_BYTES``), and the user's
#: text, ids and images ride the same line. Half the ceiling leaves the rest of
#: the frame room while still carrying a SKILL.md far larger than any
#: hand-written one. The design doc's 8-skill / 48k-token cap is NOT
#: implemented on main, so this does not pretend to be it (SKM-D4).
#:
#: WIRE bytes, meaning what ``_wire_bytes`` counts, because that is what the
#: frame limit itself measures: ``fit_request_frame`` weighs
#: ``json.dumps(frame)`` with its default ``ensure_ascii=True``. Counting raw
#: UTF-8 let this 384 KiB budget build a 1.125 MiB emoji frame that the
#: transport refused as a bare ``OversizedRequest`` — a 422 the phone could not
#: key ``skills_too_large`` on (agent review round 1, R-MINOR-1).
SKILL_BODY_BUDGET_BYTES = 384 * 1024

#: The opening tag of a rendered payload, capturing the typed line stored in
#: its ``invocation`` attribute. Anchored to the tag rather than searched
#: loosely so ordinary prose quoting the words cannot be mistaken for one.
_INVOCATION_TAG_RE = re.compile(r'<skill name="[^"]*" invocation="([^"]*)">')


def _wire_bytes(body: str) -> int:
    """How many bytes this body adds to the relay's JSON control frame.

    MEASURED THE WAY THE FRAME IS: ``fit_request_frame`` takes
    ``len(json.dumps(frame).encode())`` with the default ``ensure_ascii=True``,
    so a body is weighed in ESCAPED characters — one per ASCII letter, six per
    Latin-1/Greek/Cyrillic or CJK character, twelve per emoji (a surrogate
    pair). Raw UTF-8 bytes undercount every one of those, by up to 3x for a
    two-byte character.

    The two quote characters ``json.dumps`` wraps the string in are subtracted:
    they belong to the frame, not to this body, so summing several bodies sums
    exactly the text each one contributes.
    """
    return len(json.dumps(body)) - 2


class SkillAttachment(NamedTuple):
    """A resolved mention paired with the skill it names.

    The body is not carried here: reading it is the host's I/O, and the host
    owns the per-skill notice when a body is empty or unreadable.
    """

    mention: SkillMention
    skill: Skill


def plan_attachments(text: str, skills: Mapping[str, Skill]) -> list[SkillAttachment]:
    """Every skill this message fires, in text order, bodies not yet read.

    :func:`local_operator.sigils.skill_mentions` decides the trigger; this adds
    the two rules that need the block grammar:

    - a mention INSIDE an already-rendered block is ignored. A persisted payload
      carries the user's ``$name`` in the prose outside the block, so a second
      pass over that text would otherwise attach the body a second time;
    - a name already present in a block is not attached again, which is the same
      idempotence rule the reference resolver applies through ``typed=``.

    Empty bodies and read failures are deliberately NOT decided here: the host
    emits one notice per skill and keeps the rest, and that notice text belongs
    to the host.
    """
    blocked = skill_block_spans(text)
    # The spans are handed to the matcher so a token INSIDE a body never enters
    # its dedupe set: a pasted block quoting ``$research`` must not shadow a
    # real ``$research`` later in the message.
    mentions = skill_mentions(text, skills.keys(), skip_spans=blocked)
    if not mentions:
        return []
    already = set(skill_block_names(text))
    planned: list[SkillAttachment] = []
    for mention in mentions:
        if mention.name in already:
            continue
        planned.append(SkillAttachment(mention=mention, skill=skills[mention.name]))
    return planned


def render_skill_block(entries: Sequence[tuple[str, str]]) -> str:
    """The tail block for ``(name, body)`` pairs, in the caller's order.

    One preamble for the whole block rather than one per skill: the model needs
    the imperative once, and per-skill headers would repeat it for every body a
    message carries.

    Bodies are DEFUSED (:func:`local_operator.sigils.defuse_skill_block`) so a
    SKILL.md quoting a marker cannot close the block early; nothing else about
    the body is escaped, because the model must read it verbatim.
    """
    elements = [
        f'<skill name="{name}">\n{defuse_skill_block(body)}\n</skill>' for name, body in entries
    ]
    return (
        SKILL_BLOCK_OPEN
        + SKILL_BLOCK_JOIN
        + SKILL_BLOCK_PREAMBLE
        + SKILL_BLOCK_JOIN
        + SKILL_BLOCK_JOIN.join(elements)
        + SKILL_BLOCK_JOIN
        + SKILL_BLOCK_CLOSE
    )


class SkippedBody(NamedTuple):
    """A skill this message named whose body could not be attached.

    ``reason`` is ``"empty"`` when the resolver answered but the file holds
    nothing beyond its frontmatter, and ``"unreadable"`` when reading raised.
    The distinction exists for the HOST's notice, not for the wire: a phone is
    told the NAME and writes one sentence of its own copy.
    """

    name: str
    reason: str


class ExpandedInvocation(NamedTuple):
    """What one message's ``$name`` mentions expanded to.

    ``text`` is the WHOLE message to send — the user's text verbatim, then the
    tail block when anything attached — because that is the shape every caller
    stores as a prompt (``Session.prompt``, ``ContinuationCommand.text``). The
    live row stays the user's by construction: ``harness.rows.user_row_text``
    strips a complete block.
    """

    text: str
    attached: tuple[str, ...] = ()
    skipped: tuple[SkippedBody, ...] = ()


class SkillPayloadTooLarge(ValueError):
    """A message named more skill body than the transport may carry.

    Raised rather than degraded: silently dropping bodies would send a message
    that LOOKS invoked and is not, and the caller's refusal (422
    ``skills_too_large``, draft kept) is the only answer that lets the operator
    choose which skills to drop.
    """

    def __init__(self, size: int, budget: int) -> None:
        super().__init__(
            f"skill bodies total {size} bytes on the wire, over the {budget} byte budget"
        )
        self.size = size
        self.budget = budget


def skill_body_has_content(body: str | None) -> TypeGuard[str]:
    """Whether a resolved SKILL.md carries anything beyond its frontmatter.

    ``resolve_skill_url`` returns the file verbatim, so a stub whose YAML block
    is the entire file comes back as a NON-EMPTY string with zero instruction
    in it. A plain truthiness check therefore calls that a successful load and
    sends the model a ``<skill>`` block containing only ``name:`` and
    ``description:`` — a silent no-op.

    ONE predicate for every surface that expands ``$name``: the TUI's submit
    path and the mobile relay must agree about what "this skill has nothing to
    say" means, and two copies of this rule drifting apart is exactly the
    defect the byte-for-byte parity test exists to catch.
    """
    if not body or not body.strip():
        return False
    text = body.lstrip()
    if text.startswith("---"):
        # Drop the leading YAML block, if it is terminated. An UNterminated one
        # is malformed rather than empty, so it falls through and counts as
        # content: the user should see the skill fire and read the odd result,
        # not be told their skill is empty when it is broken.
        rest = text[3:]
        end = rest.find("\n---")
        if end != -1:
            text = rest[end + 4 :]
    return bool(text.strip())


def expand_invocation(
    text: str,
    skills: Mapping[str, Skill],
    read_body: Callable[[str], str | None],
    *,
    body_budget: int = SKILL_BODY_BUDGET_BYTES,
) -> ExpandedInvocation:
    """Attach the skill bodies this message names to the text it will send.

    The pure half of an expansion, so the two surfaces that submit a prompt
    (the TUI and the mobile relay) can be compared byte for byte: this decides
    WHICH names fire (:func:`plan_attachments`, the same trigger the composer's
    ink uses), reads each body through the caller's reader, and renders the
    frozen tail block with :func:`render_skill_block` after the user's text.

    PER-SKILL FAILURE IS THE CONTRACT: an empty or unreadable body skips THAT
    skill and the message still sends. Swallowing the request would be the
    worse half of the trade, and the caller names every skip so no row on
    screen reads like an invocation that never happened.

    ``read_body`` is the caller's I/O (``skill://`` resolution needs the
    discovered map), which is what keeps this function synchronous, pure and
    testable without a filesystem.
    """
    planned = plan_attachments(text, skills)
    if not planned:
        return ExpandedInvocation(text=text)
    entries: list[tuple[str, str]] = []
    attached: list[str] = []
    skipped: list[SkippedBody] = []
    total = 0
    for attachment in planned:
        name = attachment.skill.name
        try:
            body = read_body(name)
        except Exception:  # noqa: BLE001 — one skill's failure, never the message
            skipped.append(SkippedBody(name, "unreadable"))
            continue
        # ``None`` and "frontmatter only" are ONE answer on the wire and in the
        # TUI (both skip and both say "empty"), so the reason token matches
        # even though only one of the two is literally an empty file.
        if not skill_body_has_content(body):
            skipped.append(SkippedBody(name, "empty"))
            continue
        total += _wire_bytes(body)
        if total > body_budget:
            raise SkillPayloadTooLarge(total, body_budget)
        entries.append((name, body))
        attached.append(name)
    if not entries:
        return ExpandedInvocation(text=text, skipped=tuple(skipped))
    return ExpandedInvocation(
        text=f"{text}\n\n{render_skill_block(entries)}",
        attached=tuple(attached),
        skipped=tuple(skipped),
    )


def _unescape_attr(value: str) -> str:
    """Inverse of the legacy writer's attribute escaping.

    Legacy payloads escaped ``&``, ``"``, ``<``, ``>`` and a newline (as
    ``&#10;``). ``&amp;`` is undone last so it cannot double-undo; that is what
    keeps a literal ``&amp;lt;`` from inventing a character the operator never
    typed.
    """
    return (
        value.replace("&#10;", "\n")
        .replace("&gt;", ">")
        .replace("&lt;", "<")
        .replace("&quot;", '"')
        .replace("&amp;", "&")
    )


def typed_line_of(text: str) -> str | None:
    """The ``$skill`` line a legacy payload was built from, or ``None``.

    The read side of the legacy ``invocation`` attribute. A persisted user
    message IS the payload, and painting it verbatim would show a resumed
    conversation the whole SKILL.md body as the user's row and title the thread
    after it. Recovering the line here keeps replay showing what the live
    session showed.

    New payloads do not need this — their user text sits verbatim before the
    block and :func:`local_operator.sigils.skill_block_spans` strips only the
    block — but the reader stays forever for transcripts written before the
    tail-block shape. ``None`` for any text that is not one of those payloads,
    which is every ordinary message.
    """
    match = _INVOCATION_TAG_RE.search(text)
    return _unescape_attr(match.group(1)) if match else None
