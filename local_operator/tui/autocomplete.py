"""Slash-command matching and ranking for the input editor.

Purely sync and I/O-free by design: matching runs on every keystroke, feeds
the picker that draws under the editor, and must resolve deterministically
before Enter is dispatched. File/path completion is async work and lives
elsewhere (later); only commands rank here.

This module ranks, it does not decide: :mod:`local_operator.tui.widgets.
command_picker` owns which match is highlighted and what gets inserted. Two
places computing "the" match is how the highlighted row and the applied
command drift apart, so there is exactly one — :func:`match_commands`.

Scoring contract:

- exact match: 1000
- prefix match: 900, flat — registry order breaks ties
- fuzzy subsequence: 1..40, denser matches score higher
- otherwise 0 (no match), which includes the empty prefix: "nothing typed"
  is not a match, and the bare-``/`` menu is the picker's call to make
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, TypeVar, runtime_checkable

#: Exact / prefix tiers, with registry-order tie-break.
SCORE_EXACT = 1000
SCORE_PREFIX = 900
SCORE_FUZZY_MAX = 40


@dataclass(frozen=True)
class SlashCommand:
    """A user-facing slash command known to the app."""

    name: str
    description: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def names(self) -> tuple[str, ...]:
        """Primary name first, then aliases — order is the tie-break order."""
        return (self.name, *self.aliases)


@runtime_checkable
class Completable(Protocol):
    """What the picker needs of anything it can offer and complete.

    Exists so the ONE picker widget can present two different kinds of list —
    the command word (:class:`SlashCommand`) and a command's argument
    (:class:`ArgumentChoice`) — without a second widget, a second matcher or a
    second set of key bindings growing beside the first. The alternative was a
    provider-specific picker, which is how a codebase ends up with two lists
    that drift apart in look and in behaviour.
    """

    # Declared as read-only PROPERTIES, not annotated attributes. A bare
    # ``name: str`` in a Protocol demands a *settable* attribute, which no
    # frozen dataclass can satisfy — and both implementers here are frozen
    # (:class:`SlashCommand` and :class:`ArgumentChoice`), deliberately, because
    # a suggestion the picker is holding must not mutate under it between the
    # keystroke that ranked it and the Enter that acts on it.
    @property
    def name(self) -> str:
        """The value a completion inserts."""
        ...

    @property
    def description(self) -> str:
        """One line explaining what this is."""
        ...

    @property
    def names(self) -> tuple[str, ...]:
        """Primary name first, then aliases."""
        ...


#: Bound to :class:`Completable` so a matcher hands back the SAME concrete type
#: it was given: ranking a list of commands returns commands, ranking a list of
#: argument choices returns argument choices. Without it every caller would have
#: to narrow the result back at runtime and silently drop anything it failed to
#: recognise.
ChoiceT = TypeVar("ChoiceT", bound=Completable)


@dataclass(frozen=True)
class ArgumentChoice:
    """One value offered for a slash command's ARGUMENT.

    ``detail`` is state rather than explanation — "logged in", "needs login" —
    and is rendered right-aligned, away from ``description``, because the two
    answer different questions: what this thing IS versus where it stands right
    now. A user typing ``/logout`` is choosing by state, so putting the state in
    the description column would bury the only column they are scanning.
    """

    name: str
    description: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)
    detail: str = ""
    #: Paints ``detail`` in the danger tint when the state is a problem the user
    #: should notice (a server that failed, a credential that cannot be read).
    alert: bool = False

    @property
    def names(self) -> tuple[str, ...]:
        """Primary name first, then aliases — order is the tie-break order."""
        return (self.name, *self.aliases)


def score_command_text_match(prefix: str, target: str) -> int:
    """Score how well a typed ``prefix`` matches a command ``target``.

    Case-insensitive. Exact 1000 > prefix 900 (flat, so registration order
    breaks ties) > fuzzy subsequence 1..40 > no match 0. The fuzzy band
    rewards density: consecutive matched characters and early matches push
    the score toward 40.
    """
    lower_prefix = prefix.lower()
    lower_target = target.lower()
    if not lower_prefix:
        return 0
    if lower_prefix == lower_target:
        return SCORE_EXACT
    if lower_target.startswith(lower_prefix):
        return SCORE_PREFIX
    return _subsequence_score(lower_prefix, lower_target)


def _subsequence_score(prefix: str, target: str) -> int:
    """Score ``prefix`` as an in-order subsequence of ``target``, 1..40 or 0."""
    score = 0
    prev_index = -2
    target_index = 0
    for char in prefix:
        found = target.find(char, target_index)
        if found < 0:
            return 0
        if found == prev_index + 1:
            score += 2  # consecutive run: dense match
        else:
            score += 1
        prev_index = found
        target_index = found + 1
    if score <= 0:
        return 0
    return max(1, min(SCORE_FUZZY_MAX, score))


def match_commands(
    text_before_cursor: str, commands: list[SlashCommand]
) -> list[tuple[str, SlashCommand]]:
    """Return ``(display_name, command)`` matches for slash text, best first.

    ``text_before_cursor`` is the editor text up to the caret; matching only
    applies to a single token starting with ``/``. Ties keep registration
    order (the prefix tier is deliberately flat, so registration order breaks ties).
    """
    token = text_before_cursor.strip()
    if not token.startswith("/"):
        return []
    typed = token[1:]
    scored: list[tuple[int, int, str, SlashCommand]] = []
    for registry_index, command in enumerate(commands):
        best = 0
        best_name = command.name
        for alias_index, alias in enumerate(command.names):
            score = score_command_text_match(typed, alias)
            if score > best:
                best = score
                best_name = alias
        if best > 0:
            scored.append((-best, registry_index, best_name, command))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [(name, command) for _, _, name, command in scored]


def match_choices(query: str, choices: Sequence[ChoiceT]) -> list[tuple[str, ChoiceT]]:
    """Rank ``choices`` against a bare ``query`` token, best first.

    The argument-side counterpart to :func:`match_commands`, sharing
    :func:`score_command_text_match` so a provider and a command are ranked by
    exactly the same rules — a user who has learned that ``/lgt`` finds
    ``logout`` should find ``anthrpc`` finds ``anthropic`` without learning a
    second behaviour.

    Two deliberate differences from :func:`match_commands`. There is no leading
    ``/`` to strip, because an argument is a bare word. And the returned display
    name is ALWAYS ``choice.name``, never the alias that happened to match: a
    command's aliases are themselves typeable commands (``/models`` really runs),
    whereas an argument's aliases are only a way to FIND it — ``claude`` finds
    the ``anthropic`` provider but ``/login claude`` is not a thing. Returning
    the alias would put a word into the buffer that the command then rejects as
    unknown.

    An empty ``query`` returns everything in the given order, because "I typed
    the command and stopped" is a request to see the whole set, not a failed
    match.
    """
    if not query:
        return [(choice.name, choice) for choice in choices]
    scored: list[tuple[int, int, ChoiceT]] = []
    for order, choice in enumerate(choices):
        best = max(
            (score_command_text_match(query, alias) for alias in choice.names),
            default=0,
        )
        if best > 0:
            scored.append((-best, order, choice))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [(choice.name, choice) for _, _, choice in scored]
