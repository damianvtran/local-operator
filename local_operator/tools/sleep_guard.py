"""Refuse a foreground ``bash`` call that is mostly a long literal ``sleep``.

Why this exists
---------------
A foreground ``bash`` call cannot be interrupted by a ``hub`` NOTE. Notes ride
``Session.queue_aside``, which is delivered at the next tool-batch boundary
and wakes a parked ``wait`` — but a running ``bash`` is not a boundary, so the
note waits for the command to return. Measured on the live child session
``f7318cc06bdd`` (``backend-catalogue-paging``, 2026-09-24): it spent most of
4.4 hours in calls such as::

    sleep 1500; tail -c 250 "$LOCAL_OPERATOR_SCRATCHPAD/suite-final2.log"; echo; uptime
    sleep 1800; tail -c 200 "$LOCAL_OPERATOR_SCRATCHPAD/suite-final3.log"; ...

polling a pytest run, and every one of its parent's three notes was read only
when the sleep ran out — the last one a decision to STOP the work, delivered
after a 30-minute sleep. The desktop page showed the call as "Monitoring
suite", so nobody watching could tell it was a sleep either.

The better shape already exists: start the long work with ``background: true``
and block on it with ``wait``, which returns the moment the job settles, a
message arrives, or the session is steered. This guard is what routes the model
there — a refusal at decision time, with the replacement spelled out.

What it deliberately does not do
--------------------------------
Same posture as :mod:`local_operator.tools.search_guard`: it BLOCKS WITH A
SUGGESTION and never rewrites the command, and it errs toward allowing. A
segment is refused only when ALL of these hold:

1. it is a plain top-level ``sleep`` whose every operand is a literal duration
   (``900``, ``1.5``, ``25m``, ``1h``, ``infinity``; a ``2>/dev/null``-style
   redirection is ignored) — ``sleep "$N"``, ``sleep $((60*30))`` or anything
   else the parser cannot read as a number passes. So does ``sleep 900 2>&1``:
   the shared splitter reads that ``&`` as a background operator, and erring
   toward allowing is the posture;
2. it runs in the FOREGROUND of the shell — a segment ended by a single ``&``
   blocks nothing and passes, and so does a later pipeline stage (``x | sleep
   N``). The FIRST stage of a pipeline (``sleep 900 | cat``) is refused: the
   pipeline waits for it;
3. it sits at compound depth 0 — a sleep inside ``( … )``, ``{ … }``, or a
   ``while``/``until``/``for``/``if``/``case`` body passes, because that body
   may itself be backgrounded (``( sleep 900; notify ) &``) and a guard that
   cannot tell must not refuse;
4. the foreground top-level sleeps in the command add up to MORE than
   :data:`LONG_SLEEP_THRESHOLD_SECONDS`.

The whole check is skipped for ``background: true`` calls (the caller does
that): a sleep inside a background job holds no turn.

Escape hatch: prefix EACH long sleep with
``LOCAL_OPERATOR_ALLOW_LONG_SLEEP=1`` — read off the command itself, per
segment, exactly like the search guard's grant, so it is visible in the
transcript and never leaks from an inherited environment. Per segment is
load-bearing and is what the copy must say: a command holding two long sleeps
needs the prefix twice, and an ``export``ed value is not read at all (design
review D3).
"""

from __future__ import annotations

import os
import re

from local_operator.tools import search_guard

#: The env var an agent sets inline (``LOCAL_OPERATOR_ALLOW_LONG_SLEEP=1 sleep
#: 900``) to run a long foreground sleep as written.
ALLOW_ENV = "LOCAL_OPERATOR_ALLOW_LONG_SLEEP"

#: Seconds of top-level foreground ``sleep`` above which a call is refused.
#:
#: 120 s because it is the bash tool's own DEFAULT timeout
#: (``builtin.BASH_DEFAULT_TIMEOUT_SECONDS``): a call the harness would size as
#: "ordinary" is never refused on its sleep, while the minutes-long waits that
#: made a child unsteerable (180 s to 1800 s in the evidence above) all are.
#: Short settle-sleeps (``sleep 2`` after a kill, ``sleep 1`` in a readiness
#: loop) are nowhere near it. Deliberately NOT derived from that constant: the
#: two answer different questions, and a change to the default timeout should
#: not silently move what counts as a steering hazard.
LONG_SLEEP_THRESHOLD_SECONDS = 120.0

#: One ``sleep`` operand: a non-negative decimal with an optional GNU suffix.
#: BSD ``sleep`` on macOS takes the bare number; the suffixes are GNU's (and
#: newer BSD's). Either way the parse only has to RECOGNISE a literal duration.
_DURATION_RE = re.compile(r"^(\d+(?:\.\d*)?|\.\d+)([smhd]?)$")
#: A redirection word: optional fd, then `<`/`>`/`>>` and a target.
_REDIRECT_RE = re.compile(r"^\d*(?:>>?|<)\S+$")
_SUFFIX_SECONDS = {"": 1.0, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}

#: Words that open / close a compound command. Counted per segment to know
#: whether a ``sleep`` is at the top level of the shell. ``while``/``until``/
#: ``for``/``select``/``if`` are NOT openers: each is paired with the ``do`` or
#: ``then`` that follows it, and counting both against one ``done``/``fi``
#: would leave every command after a loop looking nested (and so unguarded).
_OPENERS = frozenset({"{", "do", "then", "case"})
_CLOSERS = frozenset({"}", "done", "fi", "esac"})


def _without_comment(words: list[str]) -> list[str]:
    """Drop a shell comment, so `sleep 121  # poll the suite` is still seen.

    A comment is a word whose FIRST character is ``#`` (bash starts one at the
    start of a word, not mid-word): everything from there on is not part of the
    command, so leaving it in makes the operand list unparsable and the sleep
    invisible. ``_words`` keeps quotes, so a ``#`` inside a quoted argument is
    not the first character of its word and survives — ``sleep "1#2"`` is
    still a literal the guard reads.
    """
    for index, word in enumerate(words):
        if word.startswith("#"):
            return words[:index]
    return words


def _duration_seconds(words: list[str]) -> float | None:
    """Total seconds of ``sleep``'s operands, or None when any is not a literal.

    ``sleep`` sums its operands (``sleep 1m 30`` is 90 s). ``infinity`` is
    GNU's spelling of forever and counts as unbounded.
    """
    if not words:
        return None
    total = 0.0
    for raw in words:
        word = search_guard._unquote(raw)
        if word == "infinity":
            return float("inf")
        m = _DURATION_RE.match(word)
        if m is None:
            return None
        total += float(m.group(1)) * _SUFFIX_SECONDS[m.group(2)]
    return total


def _depth_delta(segment: str) -> int:
    """How much this segment moves the compound-command depth.

    Parentheses are counted outside quotes and outside ``$( … )``, whose
    parentheses balance within the word; the keyword/brace openers and closers
    are counted as whole words. Approximate on purpose: the only use is "is a
    sleep at the top level?", and every error in either direction is caught by
    the fact that a mis-counted segment almost never parses as a bare sleep.
    """
    delta = 0
    for word in search_guard._words(segment):
        if word.startswith(("'", '"')):
            continue
        if word in _OPENERS:
            delta += 1
        elif word in _CLOSERS:
            delta -= 1
        # `(` / `)` at the edges of a word: `(sleep`, `echo)`, `)`. A `$(`
        # substitution opens and closes inside one word, so strip those first.
        bare = re.sub(r"\$\([^()]*\)", "", word)
        delta += len(bare) - len(bare.lstrip("(")) if bare.startswith("(") else 0
        delta -= len(bare) - len(bare.rstrip(")")) if bare.endswith(")") else 0
    return delta


def check_long_sleep(command: str, *, offered: frozenset[str] | None = None) -> str | None:
    """Return a refusal when ``command`` is dominated by a long foreground sleep.

    ``None`` means run it. The caller skips this entirely for
    ``background: true`` calls.

    ``offered`` is the reader's live tool inventory when the caller has it — the
    plan-time hook does, because ``_plan_call`` holds ``context.tools``. It
    decides which replacements the refusal may name; see :func:`_block_message`.
    """
    total = 0.0
    depth = 0
    for segment, piped, terminator in search_guard._split(command):
        at_top = depth == 0
        depth = max(depth + _depth_delta(segment), 0)
        if not at_top or piped or terminator == "&":
            # Inside a compound that may be backgrounded, a stage of a pipeline,
            # or backgrounded by the shell itself: not ours to judge.
            continue
        stripped, assigns = search_guard._strip_env_assignments(segment)
        if search_guard._truthy(assigns.get(ALLOW_ENV)):
            continue
        words = _without_comment(search_guard._words(stripped))
        if not words or os.path.basename(search_guard._unquote(words[0])) != "sleep":
            continue
        # A redirection (`2>/dev/null`, `>log`) changes nothing about how long
        # the sleep holds the call, so it must not hide one.
        seconds = _duration_seconds([w for w in words[1:] if not _REDIRECT_RE.match(w)])
        if seconds is None:
            continue
        total += seconds
    if total > LONG_SLEEP_THRESHOLD_SECONDS:
        return _block_message(total, offered)
    return None


def _format_seconds(seconds: float) -> str:
    if seconds == float("inf"):
        return "forever"
    if seconds >= 60 and seconds % 60 == 0:
        return f"{int(seconds)} s ({int(seconds // 60)} min)"
    return f"{seconds:g} s"


#: The three tools the advice can hand the waiting to, in the order it names
#: them, and the only names it may assert: a reader that does not hold one is
#: the reader this copy must not route into ``Tool not found`` (design review
#: D2 — a coder child has no ``wake``, a declared-inventory session has none of
#: the three).
_ADVICE_TOOLS = ("wait", "jobs", "wake")


def _block_message(seconds: float, offered: frozenset[str] | None = None) -> str:
    """The refusal, stated so the model can act on it rather than guess.

    Who can read this, stated as measured rather than as intended:

    * the MODEL always gets every line, uncropped — the refusal rides the tool
      result verbatim;
    * the operator sees the bullets on the failed-call card, whose advice rows
      are painted as ONE cropped row each (91 cells at a 100-column frame, 72 at
      80, 52 at 60, 41 at 50 — design review round 2's measure);
    * the operator does NOT see them on this refusal's LIVE row: a plan-time
      refusal settles on the not-run ending, whose reason is clipped at
      ``NOT_RUN_REASON_MAX_CHARS`` (200) by the producer and whose row cannot
      expand, so the live frame carries ``blocked: this c…`` at every width.
      That is a pre-existing property of the not-run surface, not of this copy,
      and it is recorded as a follow-up on the PR rather than fixed here — the
      row is a ``ToolCard``/loop change (design review D6).

    So the two constraints below shape the copy that the MODEL reads and that
    the operator reads once the card is rebuilt (a resumed session, a child's
    trajectory):

    * every bullet LEADS with its actionable token, because such a card crops
      each advice line at the widths above, and the unhedged copy put ``wait``
      at cells 86-92 and the hatch variable at 57-90, i.e. cut the two most
      useful tokens exactly on the frames operators run (design review D1);
    * bullets name ONLY tools the reader holds (``offered``), because the
      hedge-everything middle ground asks the reader to audit its own inventory:
      with none of the three it says so and leads with the hatch, the one bullet
      that works for such a reader (design review D2).

    ``offered=None`` is the caller having no inventory to hand (the direct
    ``execute`` path); that is the only case that names tools speculatively.
    """
    limit = int(LONG_SLEEP_THRESHOLD_SECONDS)
    head = (
        f"blocked: this command sleeps {_format_seconds(seconds)} in the foreground "
        f"(over {limit} s), where it is deaf to a hub note — a note from your parent "
        "or a peer waits until the sleep ends, while a steer does reach it and "
        "detaches the command to a background job. Until one of those happens the "
        "session looks busy while it does nothing."
    )
    hatch = (
        f"  - `{ALLOW_ENV}=1` prefixed on each long sleep — inline, per segment, "
        "since an exported value is not read"
    )
    if offered is not None and not offered.intersection(_ADVICE_TOOLS):
        return (
            f"{head}\n"
            # Short on purpose: this line is the one place the copy speaks in
            # prose where the others put `Do one of:`, and at 108 cells it
            # painted as `…so there is nothing he…` with the bullets' lead-in
            # swallowed at every width (design review D8). 33 cells fits the
            # 41-cell painted lane at 50 columns — the suggested wording
            # ("This session has no `wait`, `jobs` or `wake`.") measures 45 and
            # would still lose its own full stop there, so it is shortened to the
            # same claim. No sentence here is worth losing the lead-in.
            "No `wait`, `jobs` or `wake` here.\n"
            "Do one of:\n"
            f"{hatch};\n"
            "  - write the command so nothing waits: this call holds until the last "
            "long sleep in it ends, whatever else the session offers."
        )
    bullets = []
    if offered is None or "wait" in offered:
        # `wait` leads, not `background: true`: with the other order the second
        # token ended at cell 52 against a 52-cell painted lane at 60 columns, so
        # it rendered as `` then `wait… `` — an unclosed token is not the remedy
        # (design review D7).
        bullets.append(
            "  - `wait` on the job id of a `background: true` long work — it returns "
            "on the job settling, on a note arriving while it is parked, or when the "
            "wait is cancelled (size `wait_ms` in ms)"
        )
    if offered is None or "jobs" in offered:
        bullets.append(
            "  - `jobs op='peek'` (job_id, since=<seq>) to check progress instead of "
            "sleeping and tailing a log"
        )
    if offered is None or "wake" in offered:
        bullets.append("  - `wake` to check back much later")
    lines = [head, "Do one of:"]
    lines.extend(f"{bullet};" for bullet in bullets)
    lines.append(f"{hatch}.")
    return "\n".join(lines)
