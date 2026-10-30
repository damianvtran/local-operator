"""Title quality for the ``notify`` tool's items (§3.1).

The inbox list shows the TITLE ALONE, so a title that identifies no event is a
row the operator has to read and dismiss to learn nothing. The reported
population was 25 such rows across six conversations — ``x``, ``unused``,
``placeholder``, ``no-op``, ``y``, ``z`` — every one with an empty body.

The boundary drawn here is deliberately one-sided: **refuse provably
content-free titles, never judge a short-but-real one.** Blocking a legitimate
notification to catch noise is the worse failure, because the model's cheapest
way to obey is then to post nothing at all, and the operator loses the item
that mattered. So each rule below refuses only what carries no information at
all: a null word, or fewer than three characters to identify an event by.

The body is deliberately NOT consulted. It is not on the list row, so a
substantial body behind a ``x`` title still renders as a row the operator has
to open to find out what happened.

This module is called from :class:`~local_operator.tools.builtin.NotifyParams`
— the one path a model drives — and NOT from ``NotificationStore.notify``.
Server-side writers (an approval gate's item is titled with the tool name, a
test's fixture is ``"one"``) use short titles on purpose and must keep working.
"""

from __future__ import annotations

#: Whole titles that name no event. Compared after :func:`_normalise`, so
#: ``No-Op``, ``no op`` and ``noop`` are one entry; the spelling here is the
#: operator-facing vocabulary, not a matching key.
#:
#: Whole-title equality ONLY. ``"Update: 3 PRs merged"`` and ``"Tests failing on
#: shard 3"`` both contain a word from this set and both carry real information;
#: a substring or startswith test would refuse them and teach the model to
#: avoid the useful openings rather than the empty ones.
_PLACEHOLDER_TITLES = frozenset(
    {
        "x",
        "y",
        "z",
        "noop",
        "no-op",
        "n/a",
        "none",
        "null",
        "test",
        "todo",
        "tbd",
        "update",
        "unused",
        "placeholder",
        "wip",
        "done",
        "ok",
        "okay",
        "note",
        "misc",
    }
)

#: The information floor, in alphanumerics. Three is the smallest count that
#: still admits titles people really write (``"PR #9"`` → ``PR9``, ``"CI green"``
#: → ``CIgreen``) while refusing the reported one- and two-character
#: placeholders. Counting alphanumerics rather than characters is what makes the
#: rule script-independent: ``"修复完成"`` is four, ``"Ошибка деплоя"`` is eleven,
#: and ``"🚀"`` is zero.
_MIN_INFORMATIVE = 3


def _normalise(title: str) -> str:
    """The title reduced to what it actually says: alphanumerics, casefolded.

    Casefolding rather than lowercasing keeps the comparison honest for scripts
    where case mapping is not one character to one character; ``isalnum`` is
    Unicode-aware, so a CJK or Cyrillic title keeps its letters (and therefore
    its information) while punctuation, whitespace and emoji drop out.
    """
    return "".join(char for char in title.casefold() if char.isalnum())


#: The vocabulary in the only form the comparison uses (``no-op`` → ``noop``).
_PLACEHOLDER_NORMALISED = frozenset(_normalise(word) for word in _PLACEHOLDER_TITLES)


def title_defect(title: str, body: str = "") -> str | None:
    """Why ``title`` is degenerate, or ``None`` when it says something.

    ``body`` is accepted and unused: the signature states at the call site that
    the body cannot rescue a title, which is the misreading this guard exists to
    stop. Keeping the parameter is cheaper than a comment the caller would not
    read.
    """
    del body  # not on the list row; see the module docstring
    said = _normalise(title)
    if not said:
        # Whitespace, punctuation or emoji only. This currently reaches
        # ``NotificationStore.notify`` and comes back as a bare ``ValueError``
        # that the tool turns into "the inbox could not be written" — a dead end
        # naming neither the cause nor a way out.
        return "after ignoring punctuation and spaces the title says nothing at all"
    if len(said) < _MIN_INFORMATIVE:
        return (
            f"its {len(said)} letter or digit"
            f"{'' if len(said) == 1 else 's'} cannot identify an event "
            f"(at least {_MIN_INFORMATIVE} are needed)"
        )
    if said in _PLACEHOLDER_NORMALISED:
        return f"{title.strip()!r} is a placeholder word, not a statement"
    return None
