"""Manual skill invocation: the ``$name`` composer prefix.

Semantic routing (:mod:`local_operator.skills.index`) decides which skills a
turn *probably* needs. This module is the other half: the user saying which
skill they want, by name, with no embedder in the loop.

``$research fix the login bug`` loads the ``research`` SKILL.md body and hands
it to the model together with the request. Three properties are deliberate:

- **Deterministic.** The body is injected as context, not suggested as a
  ``read skill://`` the model may skip. "Manual invocation" that the model can
  decline is not invocation, it is a hint.
- **Hidden skills are reachable.** A skill with ``hide: true`` /
  ``disable-model-invocation`` is excluded from semantic routing on purpose,
  and until now was reachable only if the agent happened to read its URL.
  Naming it explicitly is exactly the "never auto-select, let me fire it"
  case that flag describes, so :func:`parse_invocation` does not filter on
  ``hide``.
- **The frozen block is untouched.** Selection is frozen per session for
  prompt-cache warmth and re-opens only on compaction
  (``session_factory._render_knowledge_block``). Injecting at the MESSAGE
  level rides the volatile tail that changes every turn anyway, so a manual
  invocation costs nothing in cache warmth on the turns around it.

The recognition rule is narrow on purpose, because ``$`` is also money and
also shell. A token is an invocation only when it is the FIRST token of the
text being submitted, and only when what follows the ``$`` matches a
DISCOVERED SKILL NAME. ``$100 for the redesign`` matches no skill and is plain
prose; so is a stray ``$`` alone. That is what keeps this from needing an
escape rule: the vocabulary decides, so nothing that is not a skill name is
ever captured.

**The position rule is enforced by the COMPOSER, not by this parser being the
only grammar there is.** The composer's picker opens on a ``$`` anywhere a word
boundary allows, and accepting a row REASSEMBLES the draft — the token moves to
the front with the surviving prose as its request, staged for the user to read
and send (``command_picker.completion_for``'s ``SKILL`` branch, and
``Editor._complete_skill``). So the user can reach a skill mid-sentence while
what arrives HERE is still a prefix. Keeping this parser anchored is therefore
not a limitation to be lifted later; it is what buys two properties that inline
matching would cost:

- A pasted document whose line 3 reads ``$research …`` never fires a skill
  nobody asked for (see ``app._expand_invocation``, which parses the TYPED line
  before pastes are spliced in for exactly this reason).
- The money / shell-variable guard stays trivially safe HERE. Because this
  parser only ever looks at offset 0, ``a $5 coffee`` and ``echo $PATH`` are
  prose by construction: no rule has to be argued about, because there is no
  position at which they could match.

Which layer enforces what is worth being exact about, since the composer now
has a grammar of its own. THIS module decides what a submitted string MEANS,
and it is anchored. The composer decides what OPENS A LIST, and it is inline —
so it carries the mid-draft half of the shell-variable guard that anchoring
gives this module for free: an inline ``$`` ranks by CASE-SENSITIVE PREFIX and
only when the typed query contains a LOWERCASE LETTER
(``command_picker.skill_suggestions``), because any match there kept the picker
open and turned the user's Enter into a completion that rewrote their draft
instead of sending it (``$LANG`` reaching a ``planning`` skill, ``$DEBUG``
reaching ``debug``, and a bare trailing ``$`` reaching the entire vocabulary).
Fuzzy and case-insensitive matching both survive at the leading position, where
a ``$`` typed first is unambiguous.

The caveat below therefore has a WIDER inline reach than "a skill named after a
variable", and it is worth being exact: inline matching is by PREFIX, so any
lowercase-containing token that prefixes any skill name matches — ``$path``
reaches ``pathfinder``, ``$lang`` reaches ``language-tutor``, and lowercase
environment variables like ``$http_proxy`` are in the same class. Accepted for
the same reason as below: the vocabulary is the user's, and no rule here can
outrank it.

The converse is worth stating plainly: matching is case-insensitive and the
vocabulary is user-controlled, so a skill actually NAMED ``path`` or ``editor``
would make ``$PATH is unset`` or ``$EDITOR is vim`` invoke it — at the LEADING
position, where this parser looks. Nothing here can tell those apart from a
deliberate invocation — the name is the whole grammar — and the transcript row
shows what was typed either way, so the user can see it happened. Naming a
skill after a common environment variable is the thing to avoid.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import NamedTuple

from local_operator.skills.discovery import Skill

#: A leading ``$name`` token. ``name`` uses the skill-name character set:
#: letters, digits, hyphen, underscore and dot, because real skill names
#: contain all of them. The consequence is that ``.`` and ``-`` do NOT end the
#: token — ``$research.`` parses the name ``research.``, misses the vocabulary
#: and is sent as prose, which is the safe direction but not a sentence that
#: invokes. ``$research,`` DOES work, since ``,`` is outside the class.
#: Anchored at the start: what is SUBMITTED is a prefix, because the composer
#: reassembles an inline ``$`` to the front before Enter (see the module
#: docstring). A ``$`` still sitting mid-draft at submit time was never engaged
#: through the picker and is prose. A ``$`` followed by nothing usable does not
#: match at all.
_INVOCATION_RE = re.compile(r"^\$([A-Za-z0-9][A-Za-z0-9._-]*)")

#: The opening tag of a rendered payload, capturing the typed line stored in
#: its ``invocation`` attribute. Anchored to the tag rather than searched
#: loosely so ordinary prose quoting the words cannot be mistaken for one.
_INVOCATION_TAG_RE = re.compile(r'<skill name="[^"]*" invocation="([^"]*)">')


class SkillInvocation(NamedTuple):
    """A resolved ``$name`` prefix and the request that followed it.

    ``request`` is the buffer with the token removed and stripped; it is
    legitimately ``""`` when the user typed only ``$name``, which means "run
    this skill, the skill itself says on what".
    """

    skill: Skill
    request: str
    #: The raw token as typed (``"$research"``), for echoing a row that shows
    #: what the user actually wrote rather than a reconstruction of it.
    token: str
    #: The whole line the user typed, stripped. Carried into the rendered
    #: payload so a RESUMED session can recover it (see :func:`typed_line_of`);
    #: without it the persisted message — which is the payload — replays as the
    #: user's row and the skill body becomes the conversation's title.
    typed: str = ""


def parse_invocation(text: str, skills: Mapping[str, Skill]) -> SkillInvocation | None:
    """Resolve a leading ``$name`` against the discovered vocabulary.

    Returns ``None`` — meaning "this is ordinary prose, send it unchanged" —
    when the text does not start with ``$``, when the token is not a known
    skill name, or when the name is followed immediately by a word character
    the pattern would have to swallow. Never raises: an unparseable draft is
    always just a draft.

    Deliberately ANCHORED even though the composer's picker is inline: the
    composer reassembles the token to the front before submit, so a ``$`` that
    is still mid-text when this runs is prose the user typed, not an invocation
    they engaged. See the module docstring for the two properties that buys.

    Matching is case-insensitive because skill names are lower-case by
    convention while a sentence-start ``$Research`` is a natural thing to type,
    and the ``skills`` mapping is keyed by the discovered name.
    """
    stripped = text.lstrip()
    match = _INVOCATION_RE.match(stripped)
    if match is None:
        return None
    name = match.group(1)
    skill = skills.get(name)
    if skill is None:
        # Case-insensitive second pass. Done as an explicit scan rather than by
        # lower-casing the mapping up front so the ORIGINAL discovery name is
        # what gets resolved and echoed — the skill is `research`, even when
        # the user typed `$Research`.
        lowered = name.lower()
        skill = next(
            (candidate for key, candidate in skills.items() if key.lower() == lowered),
            None,
        )
    if skill is None:
        return None
    request = stripped[match.end() :].strip()
    return SkillInvocation(
        skill=skill,
        request=request,
        token=match.group(0),
        typed=stripped.strip(),
    )


def render_invocation(invocation: SkillInvocation, body: str) -> str:
    """Build the message text that carries an invoked skill to the model.

    The body is delivered inside a tagged block with an imperative, mirroring
    :func:`local_operator.skills.index.render_block`'s voice: a bare paste of
    SKILL.md reads to the model as reference material it may consult, and the
    whole point of a manual invocation is that the user has already decided.

    A bare ``$name`` with no request is not given a fake one. The skill body is
    the instruction in that case, and inventing "follow this skill" text around
    it would put words in the user's mouth that the skill may contradict.

    The opening tag carries the typed line in an ``invocation`` attribute. That
    is not decoration for the model: the payload is what gets PERSISTED, so a
    resumed session replays this string as the user's row, and without the
    typed line recorded here that row (and the conversation title derived from
    it) becomes the whole skill body. :func:`typed_line_of` reads it back, and
    keeping it inside the payload means there is no parallel store that can
    disagree with the message it describes.
    """
    header = (
        f"The user invoked the `{invocation.skill.name}` skill directly. Follow it for "
        "this request. Its reference files, if any, are listed at the end of the body "
        "and are read with `skill://<name>/<path>`."
    )
    typed = _escape_attr(invocation.typed)
    parts = [
        header,
        f'<skill name="{invocation.skill.name}" invocation="{typed}">',
        body,
        "</skill>",
    ]
    if invocation.request:
        parts.append(invocation.request)
    return "\n".join(parts)


def _escape_attr(value: str) -> str:
    """Escape a typed line for an XML-ish attribute, reversibly.

    Deliberately minimal and paired with :func:`_unescape_attr`: only the two
    characters that could end the attribute or the tag. A newline becomes a
    literal ``&#10;`` so a multi-line draft cannot break the single-line tag
    the replay scanner matches.
    """
    return (
        value.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "&#10;")
    )


def _unescape_attr(value: str) -> str:
    """Inverse of :func:`_escape_attr`; ``&amp;`` last so it cannot double-undo."""
    return (
        value.replace("&#10;", "\n")
        .replace("&gt;", ">")
        .replace("&lt;", "<")
        .replace("&quot;", '"')
        .replace("&amp;", "&")
    )


def typed_line_of(text: str) -> str | None:
    """The ``$skill`` line a rendered payload was built from, or ``None``.

    The read side of the ``invocation`` attribute. Used by transcript REPLAY:
    a persisted user message is the payload, and painting it verbatim would
    show a resumed conversation the whole SKILL.md body as the user's row and
    title the thread after it. Recovering the line here keeps replay showing
    what the live session showed — the same live/replay parity the loop-prompt
    skip in ``OperatorApp._replay_history`` exists to maintain.

    ``None`` for any text that is not one of these payloads, which is every
    ordinary message, so the caller falls through to painting it verbatim.
    """
    match = _INVOCATION_TAG_RE.search(text)
    return _unescape_attr(match.group(1)) if match else None
