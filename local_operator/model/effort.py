"""Which reasoning-effort levels a model accepts, and which one it starts on.

A leaf module on purpose: three subsystems need this table and none of them
should have to import the others. ``model.configure`` derives the spec from it,
``providers.failover`` re-derives it when a fallback swaps the model out from
under a chosen level, and the TUI reads the RESULT off the spec (never this
table) so no widget carries model-name knowledge — the same division
``supports_sampling_params`` established.

Keyed on the model ID rather than the provider that fronts it, for the reason
``_NO_SAMPLING_PARAMS`` gives: ``anthropic/claude-opus-5`` through OpenRouter is
the same model with the same knob, and a provider-keyed rule would offer the
level on one route and not the other.

The levels are the providers' own, transcribed from their docs rather than
guessed, because a level the provider does not accept is an HTTP 400 on every
turn and a level it silently ignores is a status band asserting a depth of
thought that is not in force:

- Anthropic — https://platform.claude.com/docs/en/build-with-claude/effort
  (read 2026-08-11). ``output_config.effort`` ∈ low|medium|high|xhigh|max, the
  API default is ``high`` on every supporting model, and the doc states that
  sending ``high`` is exactly equivalent to omitting the parameter. ``xhigh``
  and ``max`` are NOT universal: the doc lists them per model, hence three
  ladders rather than one.
- OpenAI — https://developers.openai.com/api/docs/guides/reasoning (read
  2026-08-11) for the value set, and the per-model pages for the subsets
  (``gpt-5.4``: "Reasoning.effort supports: none (default), low, medium, high
  and xhigh").

One cost the same Anthropic page names, and this app pays: effort shapes the
rendered prompt, so changing it between turns does not preserve cached prefixes
from earlier ones. The Anthropic client writes ``cache_control`` breakpoints and
the registry prices cache reads at a tenth of input, so a mid-conversation
change re-bills the prefix uncached on the next turn. The dial ships anyway —
being unable to see or change effort was the reported problem — but a level is
worth choosing once and leaving, and that is why nothing here nudges the user to
cycle mid-task.

Families NOT here offer nothing, which is the honest answer rather than a gap:

- Google. Our ``GoogleClient`` speaks ``generateContent`` with a
  ``generationConfig``, and the named ``thinking_level`` tiers Gemini documents
  belong to the Interactions API; the shipped 2.5-series registry rows express
  thinking as a token budget instead. Offering "medium" with no name-to-wire
  mapping would put a level on the band that never reaches the request, so
  Gemini reports ``reasoning`` and ``/effort`` says there is nothing to pick.
- DeepSeek's reasoner, Kimi's thinking models, xAI. They reason at a depth the
  API does not expose as a named tier, or expose it under a key we have not
  verified against a live request. An unverified key is a 400 on a turn the
  user did not ask to risk.
"""

from __future__ import annotations

import dataclasses
import re

#: The shared vocabulary, ASCENDING. Not the cycle order — ``next_effort``
#: indexes the PER-MODEL ladder, which is what ``shift+tab`` walks and what
#: ``/effort`` prints. What this pins is that every ladder below is an ordered
#: subset of one agreed set of words, so a level means the same thing whichever
#: family it came from and the ladders can be compared rung for rung.
#:
#: ``minimal`` is in the vocabulary but in no ladder yet: the OpenAI reasoning
#: guide lists it as a possible value, and no model page this table transcribes
#: names it. It is reserved rather than removed so the day a model page does,
#: the word already sorts in the right place.
EFFORT_ORDER: tuple[str, ...] = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

#: What a bare cycle starts on when the model documents no default of its own.
#: The middle of the ladder is the only choice that is not a claim about an
#: undocumented default, and it is one press from either end. The ends are both
#: wrong for a key someone pressed to FIND the control: ``none`` silently
#: lobotomises the model, ``max`` silently multiplies the bill.
FALLBACK_START = "medium"


@dataclasses.dataclass(frozen=True)
class EffortSupport:
    """The levels one model accepts, and the level it is on when nothing is set.

    ``default`` is ``None`` when the provider documents the default per model
    snapshot rather than per family — OpenAI's is ``none`` on ``gpt-5.4`` and
    ``medium`` on ``gpt-5.5``. Seeding either one across the family would put a
    level on the status band that half the family is not running, so nothing is
    seeded and the band says ``reasoning`` until the user picks.
    """

    levels: tuple[str, ...]
    default: str | None = None


# Anthropic's three ladders (see the doc link above). Written as generation
# ranges rather than a model list, the way `_NO_SAMPLING_PARAMS` is: a tier list
# is what let `claude-fable-5` slip through the sampling rule, and this table
# would fail the same way — silently, by offering three levels to a model with
# five.
_ANTHROPIC_FULL = EffortSupport(("low", "medium", "high", "xhigh", "max"), default="high")
_ANTHROPIC_NO_XHIGH = EffortSupport(("low", "medium", "high", "max"), default="high")
_ANTHROPIC_BASE = EffortSupport(("low", "medium", "high"), default="high")

# OpenAI. `gpt-5.4`'s model page is the transcribed set; the o-series predates
# `none`/`minimal`/`xhigh` and takes the classic three.
_OPENAI_GPT5 = EffortSupport(("none", "low", "medium", "high", "xhigh"))
_OPENAI_O_SERIES = EffortSupport(("low", "medium", "high"))

#: First match wins, so the narrower Anthropic generations are listed before the
#: forward-reading ``5 and above`` arm. Patterns are ``search``ed against the
#: lowercased id so an aggregator prefix (``anthropic/claude-opus-5``,
#: ``openrouter/openai/gpt-5.4``) resolves to the same support as the bare id.
_EFFORT_TABLE: tuple[tuple[re.Pattern[str], EffortSupport], ...] = (
    # 4.5: the generation that introduced `effort`, and the ONE arm written as a
    # tier rather than a generation range. The doc lists exactly one 4.5 model as
    # supporting it — `claude-opus-4-5-20251101` — and the repo ships
    # `claude-sonnet-4-5-20250929` and `claude-haiku-4-5-20251001`, which a
    # generation range would have handed a ladder the provider never offered them
    # (an `output_config` key on every turn, unasked for). Forward-reading is the
    # right default for a generation that HAS the feature; it is the wrong one
    # for the generation where it was still being rolled out tier by tier.
    (re.compile(r"claude-opus-4-5(?!\d)"), _ANTHROPIC_BASE),
    # 4.6 and the Mythos preview took `max` but not `xhigh`.
    (re.compile(r"claude-[a-z]+-4-6(?!\d)|claude-mythos-preview"), _ANTHROPIC_NO_XHIGH),
    # 4.7/4.8 and every tier at generation 5 or above take both.
    #
    # `\d{2,3}`, NOT `\d{2,}`: an unbounded run also matches the 8-digit snapshot
    # date in Anthropic's canonical ids, so `claude-opus-4-20250514` and
    # `claude-sonnet-4-20250514` — both shipped registry rows, neither on the
    # doc's supported list — read as "generation 4.7 or later" and sent
    # `output_config: {"effort": "high"}` on every request. Three digits is
    # generous for a generation number and cannot swallow a date.
    (re.compile(r"claude-[a-z]+-4-(?:[7-9]|\d{2,3})(?!\d)"), _ANTHROPIC_FULL),
    (re.compile(r"claude-[a-z]+-(?:[5-9]|\d{2,3})(?!\d)"), _ANTHROPIC_FULL),
    (re.compile(r"gpt-(?:[5-9]|\d{2,})"), _OPENAI_GPT5),
    (re.compile(r"(?:^|[/:-])o[1-9](?:-|$)"), _OPENAI_O_SERIES),
)


def effort_support(model_id: str) -> EffortSupport | None:
    """What ``model_id`` accepts, or ``None`` when it exposes no effort knob."""
    lowered = model_id.lower()
    for pattern, support in _EFFORT_TABLE:
        if pattern.search(lowered):
            return support
    return None


def supported_efforts(model_id: str) -> tuple[str, ...]:
    """The levels ``model_id`` accepts, ascending; empty when it accepts none.

    Empty is what makes the control honest on a non-reasoning model: there is
    nothing to cycle, ``/effort`` says so, and no key reaches the wire.
    """
    support = effort_support(model_id)
    return support.levels if support is not None else ()


def default_effort(model_id: str) -> str | None:
    """The level ``model_id`` runs at when nothing is set, if it documents one.

    Seeded onto the spec so the band can state a real level from boot rather
    than the word "reasoning". Only safe because the provider says so: for
    Anthropic, sending ``high`` and sending nothing are documented as the same
    request, so the seed changes no behaviour and only stops the band from
    understating what is already happening.
    """
    support = effort_support(model_id)
    return support.default if support is not None else None


def resolve_effort(model_id: str, requested: str | None) -> str | None:
    """The level ``model_id`` should run at given a ``requested`` one.

    Kept when the model accepts it, otherwise replaced by that model's own
    default (or dropped). This is the whole guard against a level outliving the
    model it was chosen for — ``/model`` switching to a non-reasoning model, or
    a fallback chain swapping Claude for a model with a different ladder. The
    carried-over value would either 400 the turn or be discarded while the band
    went on claiming it.
    """
    support = effort_support(model_id)
    if support is None:
        return None
    if requested and requested.lower() in support.levels:
        return requested.lower()
    return support.default


def next_effort(levels: tuple[str, ...], current: str | None) -> str | None:
    """The level one cycle step above ``current``, wrapping at the top.

    ``None`` in (nothing selected) starts at :data:`FALLBACK_START` when the
    model has it, else the lowest level — never at whatever happens to sit at
    index 0, which on OpenAI is ``none``: a user pressing the key to find the
    control would have turned reasoning OFF with their first press.
    """
    if not levels:
        return None
    if current is None or current.lower() not in levels:
        return FALLBACK_START if FALLBACK_START in levels else levels[0]
    return levels[(levels.index(current.lower()) + 1) % len(levels)]
