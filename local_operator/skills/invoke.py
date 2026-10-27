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

import re
from collections.abc import Mapping, Sequence
from typing import NamedTuple

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
    "SkillAttachment",
    "plan_attachments",
    "render_skill_block",
    "skill_block_names",
    "skill_block_spans",
    "typed_line_of",
]

#: The opening tag of a rendered payload, capturing the typed line stored in
#: its ``invocation`` attribute. Anchored to the tag rather than searched
#: loosely so ordinary prose quoting the words cannot be mistaken for one.
_INVOCATION_TAG_RE = re.compile(r'<skill name="[^"]*" invocation="([^"]*)">')


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
    mentions = skill_mentions(text, skills.keys())
    if not mentions:
        return []
    inside = skill_block_spans(text)
    already = set(skill_block_names(text))
    planned: list[SkillAttachment] = []
    for mention in mentions:
        if any(start <= mention.start < end for start, end in inside):
            continue
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
