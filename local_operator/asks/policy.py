"""The queued-ask policy: one module constant, its bounds, and the timeout parse.

WHY THIS MODULE EXISTS (design ``docs/design/ask-nonblocking.md`` §3, D6/D9).
The queued, timeout-bounded ``ask`` is the SHIPPED DEFAULT: every process that
can ask queues, and the answer comes back as its own turn.

It shipped DARK for a reason that has now expired (D6): core PRs A1–D merged
behind one module constant — ``NONBLOCKING_ASK`` — so no release window could
carry a half-feature (a queue engine with no surface, or a TUI asking a backend
that still blocks). The surfaces landed, so PR F flipped the constant. It does
NOT delete the blocking path — the kill switch below still selects it, which is
why that path stays tested.

The env var (``LOP_ASK_NONBLOCKING``) is an ENV seam, deliberately not a config
key: it has to be settable by a test or an evidence capture without touching the
operator's settings store, and a user-facing setting would be a promise this
note does not make (D9: no new config key).

**This module is stdlib-only by contract**, like ``wakes/store.py``: the index
readers (a cold reader, the aggregate "all my open asks" view) reach the policy
through here, and a pydantic or session import would put those on that path.
``tests/unit/test_import_graph.py`` pins the import closure of ``asks/store.py``
(the leaf, which imports nothing of ours); this module adds only that one sibling
import, which carries stdlib and nothing else.

THE TIMEOUT IS PER ASK, NOT PER QUESTION, and its floor is two minutes so a
deadline can never be shorter than the session's own 60 s wake tick
(``harness/wake.py MAX_ARM_MS``) — a sub-tick deadline would be decided by timer
granularity rather than by the model's choice. The 24 h cap equals
``DEFAULT_UNATTENDED_GATE_TIMEOUT_H`` by coincidence, not coupling: approvals
keep ignoring asks and asks ignore ``runtime.unattended_gate_timeout`` (§3).
"""

from __future__ import annotations

import os
from typing import Any

#: How long past its deadline an ask can still be answered and ATTRIBUTED (the
#: ``late`` status), and — separately — how long a timeout notice or response is
#: still worth injecting. Past this the ask folds to ``expired`` and injects
#: nothing: it is stale by the same 7-day bound the wake supervisor already uses
#: for a wake it slept through (``wakes/supervisor.py`` STALE_AFTER_S).
#:
#: Re-exported from the leaf module (``asks/store.py``, whose fold needs it)
#: rather than defined here: a second copy is how the index and the delivery
#: path would start disagreeing about what "too old" means.
from local_operator.asks.store import LATE_WINDOW_S as LATE_WINDOW_S

#: THE DEFAULT, AND ITS KILL SWITCH. ``True`` = ``ask`` enqueues and returns a
#: receipt instead of awaiting a human — the shipped behaviour. ``False`` = the
#: blocking path, unchanged since before this feature (the §5 invariant), which
#: is what ``LOP_ASK_NONBLOCKING`` still selects as an OPERATOR-FACING ESCAPE
#: HATCH.
#:
#: THE DIRECTION IS DELIBERATE, and so is testing membership in the KILL set
#: rather than truthiness of an enable set: an ABSENT variable (or an empty one)
#: must leave the shipped default, so a queue can never be switched off by a
#: variable nobody set. Only ``0``/``false``/``no``/``off`` (case- and
#: whitespace-insensitively) turn the queue off; anything else, including a
#: typo, leaves it on. A kill switch that a typo could arm would be a defect in
#: the direction that hurts the operator it exists for.
#:
#: WHY IT FLIPPED (2026-10-03). The queue was dark to keep the §5 invariant
#: while the surfaces landed; they have landed (the TUI ask list, the desktop
#: wire, the relay/web sheet). Meanwhile the one operator who needed it was
#: hitting the old blocking behaviour daily — an ask backlog, errors on
#: answering, the same question answered twice through the TUI and the UI — so
#: the default is the queue and the blocking path is the escape hatch.
#:
#: Read from the environment ONCE, at import, because that is the only moment
#: at which a process can be said to have started in one mode or the other; a
#: test that needs the other mode monkeypatches this attribute (or sets the env
#: var before importing).
NONBLOCKING_ASK: bool = os.environ.get("LOP_ASK_NONBLOCKING", "").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}


def enabled() -> bool:
    """Whether the queued-ask path is live in this process (default: yes).

    A function rather than a direct attribute read at every call site, so
    monkeypatching ``policy.NONBLOCKING_ASK`` in a test takes effect on every
    path at once. The module-level name stays the setting; nothing else caches
    it. ``LOP_ASK_NONBLOCKING=0`` is the one supported way to make it ``False``
    in a real process; everything else about the value lives on the constant.
    """
    return NONBLOCKING_ASK


#: THE ASK GATE'S KILL SWITCH (design docs/design/ask-gate.md §2.7). ``True`` =
#: ``execute_ask`` runs one forked clearance check before the unchanged enqueue,
#: so an ask whose recommended option is plainly best is diverted instead of
#: queued. ``False`` = the gate is a zero-cost, zero-token path and an ask
#: behaves exactly as today.
#:
#: Same direction and typo discipline as the queue switch above, deliberately:
#: only ``0``/``false``/``no``/``off`` (case- and whitespace-insensitively) turn
#: the gate off; an ABSENT variable or a typo leaves the shipped default — ON —
#: because the fail-open direction for this feature is the check RUNNING (its
#: every failure path enqueues unchanged), and a typo must not silently unbuild
#: a safety property. Read from the environment ONCE, at import; a test that
#: needs the other mode monkeypatches this attribute (or sets the env var before
#: import), exactly like ``NONBLOCKING_ASK``.
ASK_GATE: bool = os.environ.get("LOP_ASK_GATE", "").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}


def gate_enabled() -> bool:
    """Whether the ask gate runs in this process (default: yes, §2.7).

    The same function-not-attribute convention as :func:`enabled`, for the same
    reason: monkeypatching ``policy.ASK_GATE`` must take effect on every path at
    once (the callable's step 1 is the only reader).
    """
    return ASK_GATE


#: The WHOLE-check bound (design §2.6): the fork is one short request reading
#: the turn's warm prefix, and a fail-open budget's wrong side costs an extra
#: ask or extra latency — never a missed decision. Initially uncalibrated and
#: flagged as such in the design (§5): the later calibration reads the request
#: ledger's ``purpose="clearance"`` latency distribution. Expiry raises
#: ``TimeoutError`` → the callable returns ``None`` → the unchanged enqueue.
GATE_TIMEOUT_S = 30.0


#: The default window a queued ask stays open (design §3: "the spec's
#: less-sensitive figure"). Because a late answer stays attributable (§2.2), a
#: short default costs a TIMEOUT NOTICE, not a lost answer.
DEFAULT_TIMEOUT_S = 3600

#: Floor: 2 minutes. At or above two wake ticks, so a deadline is never
#: sub-tick (see the module docstring).
MIN_TIMEOUT_S = 120

#: Cap: 24 hours.
MAX_TIMEOUT_S = 86400

#: At or below this the ask is URGENT — derived, never a parameter (D1), so the
#: schema that rides every request grows by one field instead of two. The timeout
#: notice for an urgent ask tells the model not to wait and to resolve the
#: question another way (a `task` subagent), which is only honest while the
#: window is short enough that "someone is expected to answer now" holds.
URGENT_MAX_S = 900

#: The most asks a session may have open at once. Non-blocking asking is free
#: for the model, which is exactly why the cap exists: a re-ask loop is a cost
#: the human pays, not the model.
OPEN_ASK_CAP = 8

#: The session's earliest-deadline timer ticks at most this often
#: (``harness/wake.py MAX_ARM_MS``): a deadline hours out re-checks the wall
#: clock every minute rather than arming one long asyncio timeout, so a
#: sleep/suspend or a clock change cannot silently skip a deadline.
MAX_TICK_MS = 60_000

#: At most this many asks are projected into the derived index / wire list
#: (design §4: "cap 20 newest; open first").
PROJECTION_CAP = 20


# ---------------------------------------------------------------------------
# The timeout parameter
# ---------------------------------------------------------------------------


def parse_timeout_param(raw: Any) -> tuple[int, str | None]:
    """``(seconds, error)`` for the tool's ``timeout`` argument.

    Three shapes, one meaning. An integer is SECONDS — the unit the schema
    documents and the unit the bounds are stated in. A duration string
    (``"30m"``, ``"2h"``, ``"1h30m"``) rides
    ``harness/wake.py parse_wake_duration``, which returns MILLISECONDS: the
    division is the whole reason this function exists rather than two call
    sites each remembering, because a missed one ships a 1000x window (the
    note's §7 asserts ``"2h"`` ⇒ 7200). ``None`` means the default.

    OUT OF RANGE IS A VALIDATION ERROR THAT NAMES THE BOUNDS, never a silent
    clamp: the model calibrates from the rejection (a clamp teaches it
    nothing and it keeps asking for an hour where it meant five minutes). The
    same rule as the tool's other validators — see ``_validation_error``.

    The duration parse is imported lazily and locally: ``harness/wake.py``
    brings pydantic and the wake DTOs with it, and this module is read by the
    index readers, which must not pay for either.
    """
    if raw is None:
        return DEFAULT_TIMEOUT_S, None
    seconds: int | None = None
    if isinstance(raw, bool):
        # ``True`` is an int in Python, and a model writing ``timeout: true``
        # means nothing we can honour. Say so rather than reading it as 1 s.
        return 0, _bounds_error(raw)
    if isinstance(raw, int):
        seconds = raw
    elif isinstance(raw, str):
        text = raw.strip()
        if text.isdigit():
            seconds = int(text)
        else:
            from local_operator.harness.wake import parse_wake_duration

            ms = parse_wake_duration(text)
            if ms is None:
                return 0, _bounds_error(raw)
            seconds = ms // 1000
    else:
        return 0, _bounds_error(raw)
    if not (MIN_TIMEOUT_S <= seconds <= MAX_TIMEOUT_S):
        return 0, _bounds_error(raw)
    return seconds, None


def _bounds_error(raw: Any) -> str:
    """The one copy of the bounds sentence, so every rejection agrees."""
    return (
        f"timeout must be between {MIN_TIMEOUT_S} and {MAX_TIMEOUT_S} seconds "
        f"(2 min to 24 h), as an integer number of seconds or a duration like "
        f'"30m"; got {raw!r}.'
    )


def is_urgent(timeout_s: int) -> bool:
    """Urgency is DERIVED, never a parameter (D1)."""
    return timeout_s <= URGENT_MAX_S


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "LATE_WINDOW_S",
    "MAX_TIMEOUT_S",
    "MAX_TICK_MS",
    "MIN_TIMEOUT_S",
    "NONBLOCKING_ASK",
    "OPEN_ASK_CAP",
    "PROJECTION_CAP",
    "URGENT_MAX_S",
    "enabled",
    "is_urgent",
    "parse_timeout_param",
]
