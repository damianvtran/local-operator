"""Synchronous slash-command completion for the input editor.

Purely sync and I/O-free by design (omp's ``trySyncSlashCompletion`` split):
the slash path runs on every keystroke and must resolve deterministically
before Enter is dispatched. File/path completion is async work and lives
elsewhere (later); only commands complete here.

Scoring ports omp's ``score_command_textMatch``:

- exact match: 1000
- prefix match: 900, flat — registry order breaks ties
- fuzzy subsequence: 1..40, denser matches score higher
- otherwise 0 (no match)
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Exact / prefix tiers, matching omp's registry-order tie-break contract.
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
    order (the prefix tier is deliberately flat, same as omp).
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


def complete_command(text_before_cursor: str, commands: list[SlashCommand]) -> str | None:
    """Return the completed ``/command`` text, or None when ambiguous/no match.

    Only completes when exactly one command matches; that keeps Tab/Enter
    deterministic without a picker widget (picker is later work).
    """
    matches = match_commands(text_before_cursor, commands)
    if len(matches) != 1:
        return None
    return f"/{matches[0][0]}"
