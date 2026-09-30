"""One notice per state change: a dedupe guard for repeating session warnings.

Why this exists: the ``session_mcp_unavailable`` card is byte-identical every
time the same server fails for the same reason, and every process boot/resume
that re-attempted a dead server appended the card AGAIN — measured live on
session ``1375449bf925`` as 96 identical ``minerva-qa`` rows over ~29 hours,
including a four-card cluster inside seven minutes ("four identical cards
stacked", 2026-09-30). The TOAST for the same event already has a per-process
latch (``McpManager._auth_toasted``); the transcript ROW had none. This guard
is the row's latch, applied at the journal write so every surface that reads
the transcript (TUI, desktop UI, mobile fold, relay) sees one notice rather
than each growing its own dedupe.

Three rules, and the third is what keeps suppression from becoming silence:

* an identical card (same subject + same fingerprint) that is still
  outstanding is suppressed — where "outstanding" includes "still shown":
  a record the caller's store no longer displays is void, not a suppression
  (see ``record_visible``);
* a CHANGED card — a different reason, or the same server failing again after
  a live recovery — is a state change and emits;
* an outstanding card older than :data:`DEFAULT_REMIND_AFTER_S` re-emits as a
  reminder, so a long-lived conversation re-surfaces a condition it may have
  scrolled past instead of the suppression lasting forever.

Visibility is part of outstandingness because suppression must never outlive
the card it stands for: a caller whose store RETIRES rows it once held (the
MCP caller's replay drops everything below its latest compaction cut) must
be able to void a record, or the next identical failure would be suppressed
with no surface showing anything. ``should_emit`` therefore takes an
optional ``record_visible`` callback and re-validates its own record through
it: a record the caller no longer shows is dropped and the durable lookup
decides — the same answer for both halves of "outstanding".

The unit of identity is the RENDERED CARD, not the reason string: the
fingerprint is taken over the exact text the row would show (see
:func:`fingerprint_text`), so two reasons that differ only past the
formatter's 200-character clip are the same card to every reader and dedupe
as the same card.

Deliberately a small standalone module rather than private session state: the
equivalence rule is testable without constructing a ``Session``, and the shape
is generic — subject, fingerprint, reminder window, live-recovery marker — so
other repeating identical session notices can adopt it later instead of a
second dedupe mechanism growing beside it. Prior art this is modelled on,
in-repo: ``McpManager._auth_toasted`` (per-process latch), ``herdr/reporter``
(emission de-dupe) and ``notifications/compose.py``'s ``dedupe_key``.

Durable state is NOT held here. ``should_emit`` accepts a ``find_previous``
callback into the caller's own store, because only the caller knows where its
notices persist — for the MCP case that is a scan of the session transcript,
which is what lets a FRESH process (the common path: every resume re-attempts
the dead server) tell what the previous process already wrote.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable

#: Default staleness window for :class:`NoticeGuard`: after this long an
#: un-superseded notice re-emits. A reminder, not an expiry — the notice is
#: still outstanding and the re-emission carries the same card — tuned for the
#: operator who is days into a long-lived conversation and may have scrolled
#: the condition away. Suppression must not become "never mentioned again".
DEFAULT_REMIND_AFTER_S: float = 24 * 3600.0

#: The reminder window the MCP-unavailable notice uses (requirement: re-flag
#: only on state change, with a long staleness re-reminder). Named per-notice
#: rather than reusing the default inline so the design round can argue ONE
#: number at ONE name, and so the value is assertable in tests.
MCP_UNAVAILABLE_REMIND_S: float = DEFAULT_REMIND_AFTER_S

#: How many hex characters of the sha256 survive into a fingerprint. 16 hex
#: characters = 64 bits; the comparison set is the notices one session holds
#: (tens, not billions), where a collision needs on the order of 2**32 entries
#: by the birthday bound — so this is a readability choice (a log line can
#: name the fingerprint), not a cryptographic one.
_FINGERPRINT_HEX_CHARS = 16


def fingerprint_text(text: str) -> str:
    """Fingerprint one notice card: a short sha256 of its exact text.

    Hashing the RENDERED text — rather than the inputs that produced it — is
    what makes the equivalence class "byte-identical card". The MCP formatter
    clips its reason at 200 characters, so two reasons sharing that prefix
    render one card and must dedupe as one card; and if the card's wording
    ever changes, what counts as a duplicate changes in the same step, with no
    second place to keep in sync.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:_FINGERPRINT_HEX_CHARS]


class NoticeGuard:
    """Suppress a repeating identical notice until its state changes.

    One guard serves one session and any number of subjects (the MCP case keys
    by server name). Per-subject state, keyed by ``subject``:

    * ``_outstanding`` — the card this guard last saw EMITTED, with when.
      Written by :meth:`note_emitted` only after the caller's write LANDS
      (recording a suppressed attempt as emitted would push the reminder
      window out past a notice nobody ever saw), read by :meth:`should_emit`,
      and dropped by it when the caller's ``record_visible`` reports the
      emission is no longer shown.
    * ``_recovered`` — the subject recovered live in THIS process since that
      emission. Set by :meth:`note_recovered`, consumed by the next emission.
      Without it, a re-failure after a live recovery would be suppressed by
      the guard's own cleared-but-persisted history: the durable lookup still
      sees the old row, and the recovery row that supersedes it is
      deliberately live-only (never persisted), so the store cannot tell the
      difference.

    The two are mutually exclusive by construction: every writer of one
    clears the other, so "outstanding and recovered" is unrepresentable.
    """

    def __init__(self, *, remind_after_s: float | None = DEFAULT_REMIND_AFTER_S) -> None:
        #: ``None`` disables the staleness reminder: an outstanding notice
        #: suppresses indefinitely until a recovery or :meth:`clear` re-arms
        #: it. Callers that want the reminder pass a window (the MCP case
        #: passes ``MCP_UNAVAILABLE_REMIND_S``).
        self._remind_after_s = remind_after_s
        self._outstanding: dict[str, tuple[str, float]] = {}
        self._recovered: set[str] = set()

    def should_emit(
        self,
        subject: str,
        fingerprint: str,
        *,
        find_previous: Callable[[str, str], float | None] | None = None,
        record_visible: Callable[[str, str], bool] | None = None,
        now: float | None = None,
    ) -> bool:
        """Whether a notice for ``subject`` carrying this card should be written.

        Sources of "already outstanding", checked in order:

        1. this guard's own record — same process, since boot (or since the
           last emission). Only trusted while the caller still SHOWS it: when
           ``record_visible`` is supplied and reports the recorded emission
           is gone (the MCP case: a compaction cut drops the row from every
           replay), the record is void, dropped, and the durable lookup below
           decides instead;
        2. ``find_previous`` — the caller's DURABLE lookup, reached when the
           guard has no record (or only a void one). Called as
           ``find_previous(subject, fingerprint)`` and must return the
           timestamp of the newest PERSISTED notice for that subject when it
           is comparable to this card, else ``None``. It is a callback
           because only the caller knows where its notices persist (for MCP:
           a transcript scan);
        3. nothing outstanding — emit.

        ``record_visible(subject, fingerprint) -> bool`` is the caller's
        answer to "does your store still show the emission this guard
        recorded?". Pass it when the caller's store can RETIRE rows it once
        held (a replay with a compaction cut); ``None`` keeps the record
        authoritative, which is correct for an append-only store.

        A match against either source is suppressed only while it is FRESH;
        past the reminder window it re-emits (the same card, one more time),
        and a non-matching card always emits — that is the state change this
        guard exists to keep visible.

        The live-recovery marker short-circuits (2): after a recovery, any
        re-failure is a new state regardless of what the durable store still
        shows, and the marker is consumed by the emission's
        :meth:`note_emitted`.

        A suppression is a read. The one write here is dropping a VOID
        record — it can never suppress again, and forgetting it cannot
        invent an emission. No other state changes: only
        :meth:`note_emitted` and :meth:`note_recovered` record emissions, so
        a caller that decides not to write after an approving answer cannot
        leave the guard believing it did.
        """
        current = time.time() if now is None else now
        record = self._outstanding.get(subject)
        if record is not None:
            emitted_fingerprint, emitted_at = record
            if emitted_fingerprint != fingerprint:
                # A changed card is a state change — a new reason, or a new
                # failure after the guard was re-armed in a way that kept the
                # record. Emit.
                return True
            if record_visible is None or record_visible(subject, fingerprint):
                return self._is_stale(current, emitted_at)
            # The record is VOID — the caller's store no longer shows the
            # emission it stands for (for MCP: a compaction cut dropped the
            # row from the replay). Clear it and fall through: a record that
            # cannot be shown must not keep suppressing, and the durable
            # lookup below is the one that knows what the store still shows.
            self._outstanding.pop(subject, None)
        if subject in self._recovered:
            return True
        if find_previous is not None:
            previous = find_previous(subject, fingerprint)
            if previous is not None and not self._is_stale(current, previous):
                return False
        return True

    def note_emitted(self, subject: str, fingerprint: str, *, now: float | None = None) -> None:
        """Record a successful emission of this card for ``subject``.

        Call AFTER the caller's write lands — the timestamp is what the
        reminder window is measured from, and an emission that never happened
        must not move it. Consumes the live-recovery marker: the emission is
        now the subject's newest state, and the next identical repeat is a
        duplicate of THIS card.
        """
        self._outstanding[subject] = (
            fingerprint,
            time.time() if now is None else now,
        )
        self._recovered.discard(subject)

    def note_recovered(self, subject: str) -> None:
        """A live recovery supersedes this subject's notice: clear, and mark.

        Clearing is the direct requirement (recovered → fails again = a new
        notice); the marker is what lets the next failure EMIT even though the
        durable store still shows the pre-recovery row, because the recovery
        itself is deliberately never persisted (it asserts a process-scoped
        capability — see ``Session.journal_mcp_recovery``).

        The cross-process blind spot this leaves, stated rather than hidden: a
        recovery that happened in a PREVIOUS process is invisible here, so an
        identical re-failure after a resume stays suppressed until the
        reminder window passes. Closing it would need a persisted recovery
        record or a new state file — both rejected where ``journal_mcp_recovery``
        documents why — and the cost is bounded by the 24 h reminder.
        """
        self._outstanding.pop(subject, None)
        self._recovered.add(subject)

    def clear(self, subject: str | None = None) -> None:
        """Drop the guard's memory of ``subject`` — or of every subject.

        The re-arm hatch for tests and for a caller that knows better than the
        guard does (a session reset, a manual re-announce). Not used by the
        MCP journal path itself, which re-arms through ``note_recovered`` so
        the durable-store blind spot is handled in one place.
        """
        if subject is None:
            self._outstanding.clear()
            self._recovered.clear()
            return
        self._outstanding.pop(subject, None)
        self._recovered.discard(subject)

    def _is_stale(self, now: float, emitted_at: float) -> bool:
        """Whether an outstanding notice is old enough to re-remind.

        ``>=``, not ``>``: the window is a promise about the longest silence
        ("re-surfaces at least every 24 h"), so the boundary itself re-emits.
        """
        return self._remind_after_s is not None and now - emitted_at >= self._remind_after_s
