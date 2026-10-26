"""In-stream degeneration guard: stop a turn whose stream has become a verbatim loop.

WHY THIS EXISTS. A model's stream can collapse into one short passage repeated
verbatim until the provider's output limit ends it. Measured 2026-10-26 on an
OSWorld episode (``osworld-task_063``, session ``ep-de966d3c95c2``): three model
calls in a row each spent the full 131,072-token output budget, streaming
225-253 K characters of reasoning that ended as ``"Hmm! Hmm, hmm: hmm, hmm,
hmm. Hmm.\\n\\n"`` repeated (smallest periods of the tail: 36, 333 and 450
characters), no visible text, no tool call, ``stop_reason=length``. The loop's
empty-length handling only saw the turn AFTER the provider had billed all of it,
then retried twice more at lower effort, and each retry looped the same way.
Every surface (TUI, desktop, mobile, exec, SDK) shares that loop, so any model
on any surface can pay this.

WHAT IT DETECTS, AND WHAT IT MUST NOT. A trip requires the LAST
:data:`WINDOW_CHARS` characters of ONE channel (reasoning or visible text) to be
exactly periodic with a period of at most :data:`MAX_PERIOD_CHARS`. Because the
window is ``MAX_PERIOD_CHARS * MIN_REPEATS`` long, a trip always means at least
:data:`MIN_REPEATS` verbatim copies of the same unit, end to end, with no
variation at all. Ordinary long output never satisfies that: a table, a list or
code with repeated STRUCTURE varies in its content (a row value, an item number,
an identifier), and a single differing character anywhere in the window breaks
exact periodicity. The cost of the strictness is the one class it cannot tell
apart: output the user genuinely asked to be repeated verbatim more than ~10 KB
in a row. That is accepted; the alternative is billing the whole budget on
every loop.

COST. ``feed`` only appends to a pending list until :data:`CHECK_STRIDE_CHARS`
characters have accumulated, then folds them into a bounded tail and runs one
periodicity check over it. Memory is bounded by ``WINDOW_CHARS +
CHECK_STRIDE_CHARS`` plus one delta; work is a few window-sized C-level string
operations per stride, i.e. O(1) amortised per character, and a stream shorter
than the window never runs a check at all. Measured on the trace above, replaying
each 225-253 K character turn through the guard took ~2-3 ms of CPU in total.

STDLIB ONLY, like ``chrome_prompts``: the loop imports it on every turn and it
needs nothing from the rest of the package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: Longest repeating unit the guard looks for, in characters. The measured loops
#: had periods of 36, 333 and 450; 512 covers them with margin. A longer unit is
#: closer to "the model re-wrote the same paragraph", which is rarer, costs less
#: per repeat to notice by other means, and widens the class of legitimate
#: repeated blocks that could trip.
MAX_PERIOD_CHARS = 512

#: Minimum number of verbatim copies a trip implies. Twenty identical copies of a
#: unit, end to end, is not something varied reasoning, code, tables or lists
#: produce; the measured loop sat at hundreds of copies when the budget ran out.
MIN_REPEATS = 20

#: Tail window that must be exactly periodic. Derived, not chosen, so the two
#: constants above can never drift apart: at the longest period the window holds
#: exactly ``MIN_REPEATS`` copies, and every shorter period holds more. 10 KB
#: (roughly 2.5 K tokens) of pure repetition is the most the guard lets a loop
#: spend before it stops the stream -- against the 131 K-token budget the
#: measured turns spent.
WINDOW_CHARS = MAX_PERIOD_CHARS * MIN_REPEATS

#: How many new characters accumulate between checks. Checking per delta would
#: re-scan the window for every 2-character fragment (the measured stream
#: averaged 1.9 characters per delta); a 1 KB stride bounds the overshoot past
#: the window to one stride and keeps the check off the per-token path.
CHECK_STRIDE_CHARS = 1024

#: Length of the tail probe used to find candidate periods. Any period ``p`` of
#: the window makes the last ``_PROBE_CHARS`` characters reappear exactly ``p``
#: characters earlier, so only those offsets need a full comparison; in
#: non-repeating text a 64-character exact recurrence within 512 characters is
#: rare, which keeps the candidate list near empty.
_PROBE_CHARS = 64

Channel = Literal["reasoning", "text"]


@dataclass(frozen=True)
class DegenerationTrip:
    """What tripped the guard: enough to diagnose the stop without the stream.

    ``chars`` counts every character the channel streamed up to the trip, so a
    reader can compare it with the output budget the turn did NOT spend.
    """

    channel: Channel
    period: int
    repeats: int
    chars: int
    #: The start of the repeating unit, for the developer log only. Never shown
    #: to the user and never sent to a provider.
    sample: str

    def payload(self) -> dict[str, int | str]:
        """The transcript record (``provider_payload["degenerate_stream"]``).

        Deliberately excludes ``sample``: the payload is persisted with the
        message, and the loop's text has no business riding into a record that
        outlives the log line.
        """
        return {
            "channel": self.channel,
            "period": self.period,
            "repeats": self.repeats,
            "chars": self.chars,
        }


class StreamDegenerationGuard:
    """Watches one channel's deltas and reports the first verbatim loop.

    One instance per channel per model call: reasoning and visible text are
    watched separately because a loop in one says nothing about the other, and
    interleaving them would let a varied answer mask a looping thought (or the
    reverse). Once tripped, ``feed`` keeps returning ``None``: the caller acts
    on the first trip and the stream is being cancelled.
    """

    __slots__ = ("channel", "_tail", "_pending", "_pending_chars", "_chars", "_tripped")

    def __init__(self, channel: Channel) -> None:
        self.channel: Channel = channel
        self._tail = ""
        self._pending: list[str] = []
        self._pending_chars = 0
        self._chars = 0
        self._tripped = False

    def feed(self, delta: str) -> DegenerationTrip | None:
        """Account for one streamed fragment; return a trip the first time one fires."""
        if self._tripped or not delta:
            return None
        self._pending.append(delta)
        self._pending_chars += len(delta)
        self._chars += len(delta)
        if self._pending_chars < CHECK_STRIDE_CHARS:
            return None
        tail = self._tail + "".join(self._pending)
        self._tail = tail[-WINDOW_CHARS:]
        self._pending = []
        self._pending_chars = 0
        if len(self._tail) < WINDOW_CHARS:
            return None
        period = _smallest_period(self._tail)
        if period is None:
            return None
        self._tripped = True
        return DegenerationTrip(
            channel=self.channel,
            period=period,
            repeats=len(self._tail) // period,
            chars=self._chars,
            sample=self._tail[-period:][:80],
        )


def _smallest_period(window: str) -> int | None:
    """Smallest ``p <= MAX_PERIOD_CHARS`` with ``window[i] == window[i + p]`` everywhere.

    Candidates come from recurrences of the window's last ``_PROBE_CHARS``
    characters inside the preceding ``MAX_PERIOD_CHARS``, nearest first, so the
    first candidate that verifies is the smallest period. Verification is one
    slice comparison (``window[p:] == window[:-p]``), which is exactly the
    definition of period ``p`` and fails at the first differing character.
    """
    n = len(window)
    probe_start = n - _PROBE_CHARS
    probe = window[probe_start:]
    lowest = max(0, probe_start - MAX_PERIOD_CHARS)
    # ``rfind(sub, lo, hi)`` matches only inside ``window[lo:hi]``, so ``hi``
    # of ``n - 1`` admits a match starting at ``probe_start - 1`` (period 1) and
    # excludes the probe itself (period 0).
    end = n - 1
    while True:
        found = window.rfind(probe, lowest, end)
        if found < 0:
            return None
        period = probe_start - found
        if window[period:] == window[:-period]:
            return period
        # Next candidate must start strictly before this one.
        end = found + _PROBE_CHARS - 1
