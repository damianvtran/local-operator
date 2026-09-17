"""Immutable sidebar summaries; never prepare a transcript from a paint or click.

Names and runtime marks use the same sources as /resume. Attention is supplied
by the shared completion authority, not inferred from transcript timestamps.
The catalog has no acknowledgement path: listing a conversation is not reading it.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_operator.info.model import format_duration
from local_operator.resume import SessionRow
from local_operator.session.creation import session_category, session_created_at

logger = logging.getLogger(__name__)

#: The words for a row whose owner has stopped reporting (``live_state ==
#: "wedged"``), optionally followed by the measured age.
#:
#: A constant rather than a literal in :meth:`CatalogEntry.status` because a
#: SECOND home for it already exists: the sidebar's glyph/description pairing
#: tests name every phrase the glyphs may carry, and a copy of this string there
#: is a copy that drifts. ``/info`` and the wake surfaces paraphrase the same
#: fact in their own sentences — those are sentences, not labels, and they are
#: asserted where they are built.
WEDGED_STATUS = "Not answering · process alive"


@dataclass(frozen=True)
class CatalogEntry:
    row: SessionRow
    unseen: bool = False
    completion_kind: str = ""
    #: The token and anchor of this row's latest completion, straight from the
    #: attention store. Carried because `unseen` alone is a LEVEL — true on
    #: every poll until the session is read — and an observer that wants to
    #: announce a background session's completion needs the identity of the
    #: specific event to arbitrate on (`AttentionStore.claim_delivery`).
    #: Defaulted so every existing construction site, and the sidebar tests
    #: that build entries positionally, keep working unchanged.
    completion_token: str = ""
    anchor_id: str = ""
    #: WHY the outcome is an error, when the store carries a reason — the
    #: harness-authored cause sentence for a cut-off, or the provider's own
    #: message for a provider error. Appended to the sidebar's two error
    #: spellings so a row reports a cause instead of only a class. Additive and
    #: defaulted, so every existing construction site (including the positional
    #: ones in the sidebar tests) keeps working unchanged.
    completion_reason: str = ""
    #: Set ONLY on rows from the hidden subagent population.
    #:
    #: Load-bearing for SECTIONING, not decoration: `active` is
    #: `pending or unseen or live_state`, and 45% of subagent directories carry
    #: an unseen attention receipt, so without this flag they sort into ACTIVE
    #: SESSIONS above the user's own work.
    subagent: bool = False
    #: `origin.json`'s `agent` (role) and `label` (task label), read only for
    #: the capped page. Empty for every main row.
    agent: str = ""
    label: str = ""

    @property
    def id(self) -> str:
        return self.row.id

    @property
    def sub_title(self) -> str:
        """`label · role`, degrading to whichever half exists, else the name."""
        if self.label and self.agent:
            return f"{self.label} · {self.agent}"
        return self.label or self.agent or self.row.name

    @property
    def rank(self) -> tuple[int, int, float, str]:
        """``(tier, wake_rank, -birth, id)`` — a wake orders the PREVIOUS group only.

        ``wake_rank`` is scoped to cold rows (``not self.active``): inside
        Previous an armed wake leads, then a dormant one, then everything else.
        Every ACTIVE row gets the same constant, so the key cannot reorder
        anything the user is currently working with.

        **Why scoped rather than uniform.** The first cut applied the key in
        every tier, on the reasoning that it "only breaks ties within one
        category" and so was free. That premise is empirically false, and three
        reviewers independently reproduced the consequences: rows inside a tier
        are NOT interchangeable, because a tier mixes states that
        :func:`~local_operator.tui.widgets.session_picker.row_state_mark`
        deliberately ranks against each other.

        * tier 5 mixes ``attached`` with ``idle``, so an idle session owning a
          timer sorted above the user's own ATTACHED current session — order
          contradicting the glyph ladder, where ``○`` means *a terminal is
          watching this session* and answers "where am I?".
        * tier 4 mixes ``busy`` with ``wedged``, so a WEDGED (broken) session
          needing a person sank below a merely-busy neighbour with a wake.
        * tier 0 mixes the approval and answer gates, so an armed pending row
          displaced an OLDER pending one — and note this third case defeats a
          guard written as "presence outranks a wake", because a pending row can
          carry an empty ``live_state``.

        Scoping to Previous removes all three at the root instead of enumerating
        the states to dodge. That distinction is the real lesson: an enumerated
        guard has to be updated every time the ``live_state`` vocabulary grows,
        and the tier-0 case above is exactly what such a guard silently misses.

        It is also what the operator actually asked for, which was scoped from
        the start — "under Previous Sessions ... sorted to the top". Tiers 0/4/5
        were never in scope, and reordering them is unrequested behaviour change.

        **Glyph/order parity, stated precisely.** In Active, order defers
        entirely to the existing precedence in ``row_state_mark``; this key
        abstains. In Previous every row is cold, so ``row_state_mark`` paints
        only the wake glyph or nothing at all — the wake is the sole
        forward-looking fact a row can carry, so it leads.

        **Dormant ranks below armed, not nowhere.** ``wakes_dormant`` means the
        session was deliberately stopped and the schedule will never fire, so it
        stays under every armed row: floating it would advertise a future that is
        not coming. It still gets its own band so that every clock glyph in the
        group is CONTIGUOUS. The dim-vs-muted separation between an armed ``◷``
        and a dormant one measures 1.77:1 at the 8x17px cell this UI renders —
        below any discrimination threshold — so a dormant row stranded among the
        plain rows reads as a broken sort rather than as a distinct state.
        Banding it directly under the armed rows puts the block boundary where
        the glyph changes, and costs no row and no chrome.

        The key lives HERE rather than in :func:`session_category` because that
        function is shared with the mobile daemon, whose summaries carry no wake
        data at all (durable rows come from ``recent_session_rows``, not
        ``decorate_rows``, so ``wakes`` is always 0 there). Moving the key into
        the shared categoriser would either be a no-op on mobile or force wake
        plumbing into the daemon; keeping it in the catalog leaves both surfaces
        agreeing on the tier, which is the partition they actually share.

        Stability: a wake is as durable a fact as ``live_state`` and ``pending``,
        and shares their best-effort read — ``decorate_rows`` zeroes ``wakes``
        for a poll whose wake-index read raises, exactly as it empties the live
        map when the registry scan raises, so a transient failure can bounce a
        row for one poll and put it back on the next. That tolerance is
        deliberate (a picker that cannot read either source still lists every
        session), and this key inherits it rather than adding a new fragility.
        What IS new is an asymmetry worth naming: a cold Previous row previously
        had no poll-varying ordering input at all, and now has one. When a wake
        genuinely fires the session becomes live and changes tier, which is a
        real state change rather than churn.

        One consequence reaches MEMBERSHIP, not just order: :func:`load_catalog`
        ranks before applying ``[:limit]``, so at the ``CATALOG_SCAN_LIMIT``
        boundary an ancient session carrying a wake — armed or dormant, since
        both bands outrank a plain row — can now enter the window and displace
        a newer row that would otherwise have made it (measured at 251
        sessions). That is arguably the point of the feature — a scheduled
        session is usually an old one, and being unfindable is the report — but
        it is a real behaviour change beyond reordering, so it is recorded here.
        """
        tier = session_category(
            pending=bool(self.row.pending),
            busy=self.row.live_state in ("busy", "wedged"),
            unseen=self.unseen,
            kind=self.completion_kind,
            live=bool(self.row.live_state),
        )
        # Cold rows only. An active row takes the constant, which is what makes
        # the key structurally unable to reorder Active rather than merely
        # declining to today.
        armed = not self.active and bool(self.row.wakes) and not self.row.wakes_dormant
        dormant = not self.active and bool(self.row.wakes) and self.row.wakes_dormant
        wake_rank = 0 if armed else 1 if dormant else 2
        # Activity may update ages and badges, but must not move a click target.
        return tier, wake_rank, -self.row.created_at, self.id

    @property
    def active(self) -> bool:
        """Section membership is independent of the number of ordering categories."""
        return bool(self.row.pending or self.unseen or self.row.live_state)

    @property
    def status_code(self) -> str:
        """Stable transport spelling of the same precedence used by ``status``."""
        if self.row.pending:
            return "approval" if self.row.pending == "approval" else "answer"
        if self.row.live_state in {"wedged", "busy"}:
            return self.row.live_state
        if self.shows_completion_mark:
            return {"error": "error", "interrupted": "interrupted"}.get(
                self.completion_kind, "complete"
            )
        if self.row.live_state == "attached":
            return "attached"
        if self.row.wakes and not self.row.wakes_dormant:
            return "scheduled"
        if self.row.live_state == "idle":
            return "idle"
        if self.row.wakes:
            return "dormant"
        # A receipt is evidence of an outcome even after viewing; no receipt is
        # not evidence of success. Never turn an old unknown transcript green.
        if self.completion_token:
            return {"error": "error", "interrupted": "interrupted"}.get(
                self.completion_kind, "complete"
            )
        return "recent"

    @property
    def shows_completion_mark(self) -> bool:
        """Does an unread completion win the glyph, or does live state?

        THE single arbiter for that question: the sidebar reads it to decide
        whether to override :func:`row_state_mark`, and :attr:`status` reads it
        to decide whether to say "Unseen …". They used to make the decision
        separately, in matching order, held together by a comment asking the
        next author to keep them in step. That is what broke — and the pairing
        invariant this file already treats as load-bearing deserves a predicate
        rather than a promise.

        ``unseen`` is a LEVEL, not an edge: it is true from the moment a turn
        completes until somebody READS that session, and resuming a session
        does not acknowledge it. So the mark alone cannot be allowed to win —
        it says what the session did LAST, and the states below say what it is
        doing NOW:

        * ``pending`` — a parked gate. Already outranked unseen, and still
          does: a person is blocked on this row right now.
        * ``wedged`` — not answering NOW. A stale mark from a turn that did
          finish must not hide a runtime that has since stopped reporting, and
          a stale beat must not be read as death either: the words below say
          what was measured (``heartbeat_age_s``) rather than what it might
          mean.
        * ``busy`` — the reported bug. The session is working; painting the
          previous turn's outcome over its own spinner made seven resumed,
          healthy sessions read as seven failures.

        THE COST OF THE ``busy`` RULE, stated plainly because it is a real
        trade and not a free win: it suppresses an unread ``error`` exactly as
        it suppresses an unread ``interrupted``. A turn that FAILED, was
        resumed without being read, and is now running shows a spinner and a
        :attr:`status` of "Working" — the failure has no residue anywhere on
        the row, and the only route back to it is opening the session.

        That is accepted here, deliberately, on three grounds:

        1. Nothing is destroyed. This is a pure function of CURRENT state, not
           a latch: the instant the row stops being ``busy`` the ``✗`` and
           "Unseen error" come back, because ``unseen`` stays true until
           somebody actually reads the session.
        2. The row stays in "Active Sessions": membership is independent of
           ranking. While busy or wedged it uses the in-progress category, below
           unviewed completed outcomes; when it stops, its unread outcome earns
           that outcome's category again. Creation time orders each category.
        3. The alternative IS the reported bug, one class down. Painting an
           error mark over a running session's spinner is the same lie about
           the same row — "this is broken" over a session that is working.

        So the failure mode being accepted (a genuine error is quiet while its
        session runs, and returns when it stops) is strictly milder and strictly
        shorter-lived than the one being fixed (every resumed session claims to
        have failed, indefinitely, until read). If that trade ever needs
        revisiting, the cheap remedy is INK rather than shape — keep the
        spinner, tint it ``danger`` when ``completion_kind == "error"``, or
        widen the tooltip to "Working (last turn failed, unread)". Both are new
        design surface and belong in a round of their own, not here.

        Everything BELOW stays outranked by the mark, deliberately. ``attached``,
        an armed wake and ``idle`` are all facts about residency — true of a
        session that is merely sitting there — while "this finished and you
        have not read it" is a fact about work that is waiting for the
        operator. An unseen completion on a now-idle session is exactly the
        information the sidebar exists to keep, so it is kept.

        Ranking also suppresses stale completions while busy or wedged, so
        a row's category agrees with the live state its glyph communicates.
        """
        return (
            self.unseen and not self.row.pending and self.row.live_state not in ("wedged", "busy")
        )

    @property
    def status(self) -> str:
        if self.row.pending:
            return "Approval needed" if self.row.pending == "approval" else "Answer needed"
        # BEFORE the unseen branch, and mirrored by `shows_completion_mark`,
        # which is what the sidebar suppresses the mark on. A row that is
        # wedged or busy describes itself by what it is doing now.
        #
        # "Not answering", NOT "Not responding": the first is what the evidence
        # supports (a beat the owner's own loop stopped writing for longer than
        # ``HEARTBEAT_TIMEOUT_S``) and the second reads as a verdict on the
        # process. The qualifier carries the two facts that stop a reader
        # inferring death — the pid is still there, and here is the measured
        # age — because ``registry.classify`` is explicit that a stale beat
        # does not establish that the process stopped executing. This is also
        # the desktop catalogue's ``status.label``, so the same sentence
        # travels to the app unchanged.
        if self.row.live_state == "wedged":
            age = self.row.heartbeat_age_s
            measured = f" (last heartbeat {format_duration(age)} ago)" if age is not None else ""
            return f"{WEDGED_STATUS}{measured}"
        # THE DRAIN OUTRANKS "Working", and says it in the record's own words.
        # A draining runtime IS working, so "Working" was true and useless: the
        # one thing a reader needs off this row is that the runtime has already
        # committed to leaving and is finishing the turn first — the state that
        # makes a plain stop destructive. The phrase is the same one `lop
        # sessions` prints and `/info` shows, and the same commit that sends the
        # ``draining`` flag to the app writes it (``announce_retiring``), so the
        # catalogue and the app cannot disagree about whether this row is
        # draining (UX round 2, U8; PR #1108 reconciliation).
        if self.row.leaving:
            return self.row.leaving
        if self.row.live_state == "busy":
            return "Working"
        if self.shows_completion_mark:
            if self.completion_kind == "error":
                return self._error_label("Unseen error")
            if self.completion_kind == "interrupted":
                return self._stop_label("Unseen interruption")
            return {"interrupted": "Unseen interruption"}.get(
                self.completion_kind, "Unseen completion"
            )
        # Follows ``row_state_mark``'s precedence EXACTLY, so the tooltip can
        # never name a different state from the glyph beside it. The glyph is a
        # single character and the description is where a user finds out what it
        # meant, so the two disagreeing is worse than either being terse.
        #
        # Every branch below mirrors one in ``row_state_mark``, in its order:
        # attached outranks an armed wake, which outranks idle, and a DORMANT
        # wake falls through to the cold case — where it is still the glyph, so
        # it must still be the words (round 1, D1: a cold row with a stopped
        # schedule drew the wake mark while the tooltip said "Recent").
        if self.row.live_state == "attached":
            return "Open"
        if self.row.wakes and not self.row.wakes_dormant:
            count = self.row.wakes
            return f"Scheduled ({count} wake{'s' if count != 1 else ''})"
        if self.row.live_state == "idle":
            # An exec run is named by what it IS rather than by "Ready", which
            # invites the user to treat a supervisor's one-shot as their own
            # idle conversation. Only at THIS rung: a busy or needs-you exec run
            # says so above, because what it is doing outranks what kind it is —
            # the same precedence ``row_state_mark`` follows, so the words and
            # the glyph still cannot disagree.
            return "Running headless (exec)" if self.row.kind == "exec" else "Ready"
        if self.row.wakes:
            # Dormant: the schedule exists but the session was stopped, so it is
            # not going to fire. Named rather than hidden — a user who sees the
            # glyph needs to know why it is not going to act.
            #
            # "dormant" rather than a fresh adjective, because the stop receipt
            # this state comes FROM already says "N wakes dormant until you
            # reopen it" (``control.py`` / ``app.py``). Design review round 2
            # (D5) flagged that an earlier draft here said "paused" and split
            # the vocabulary for one fact across two surfaces; the receipt's
            # word is the established one, so this follows it rather than
            # asking the receipt to move.
            count = self.row.wakes
            return f"Stopped ({count} wake{'s' if count != 1 else ''} dormant)"
        if self.completion_token:
            if self.completion_kind == "error":
                return self._error_label("Error")
            if self.completion_kind == "interrupted":
                return self._stop_label("Interrupted")
            return {"interrupted": "Interrupted"}.get(self.completion_kind, "Complete")
        return "Recent"

    def _stop_label(self, base: str) -> str:
        """``Interrupted``/``Unseen interruption`` plus the rung, when it escalated.

        The half of design round 1's D1 that the classification change left
        open. The kind was already right — a stop the operator asked for paints
        ``⊘`` in the interrupted ink rather than ``✗`` — but the reason reached
        no surface, so a rung-3 SIGKILL and a rung-1 request produced
        byte-identical rows and the tooltip said nothing the operator could act
        on. :func:`incidents.stop_rung_phrase` is the phrase and it is empty for
        the plain request rung, deliberately: this row's own word already says
        the user stopped it, and ``/stop`` is exactly what rung 1 is.

        The same budget rule as :meth:`_error_label` — one logical line, no
        parenthetical stack — and the same tolerance: an empty reason returns
        the spelling BYTE-IDENTICAL to today's, so every pre-taxonomy record and
        every plain stop renders exactly as it always has.
        """
        from local_operator.incidents import stop_rung_phrase

        phrase = stop_rung_phrase(self.completion_reason or "")
        return f"{base} — {phrase}" if phrase else base

    def _error_label(self, base: str) -> str:
        """``Error``/``Unseen error`` plus the reason, when there is one.

        The ONE-LINE BUDGET IS THE ROW, not the tooltip, and the earlier wording
        here said the opposite (design round 1, D6, which measured it): the
        ``Tooltip`` widget WRAPS — an error row renders 36×5 and 36×6 cells and
        the longest sentence winds over four rows above the id line — so a
        claim that the tooltip truncates was the justification for dropping a
        parenthetical that would in fact fit. What the sidebar cannot afford is
        the ROW's single description cell, and what the reason is trimmed to is
        therefore its first SENTENCE: the parenthetical DETAIL
        (``runtime-retired`` carries a build pair — ``(0.54.11@b133eba →
        0.54.12@402af7f)`` — which is useful in the incident card the model
        reads and noise in a list of sessions) is dropped for brevity, not for
        clipping. An empty reason leaves the spelling BYTE-IDENTICAL to today's,
        which is what keeps every pre-taxonomy record and every completion
        unchanged.
        """
        reason = (
            self.completion_reason.strip().splitlines()[0].strip()
            if self.completion_reason.strip()
            else ""
        )
        if not reason:
            return base
        sentence = reason.split(" (", 1)[0].rstrip()
        if len(sentence) > 160:
            # A provider's own message can be a paragraph; the tooltip cannot
            # grow a second line for it.
            sentence = sentence[:157].rstrip() + "…"
        return f"{base} — {sentence}"


def entry_for(row: SessionRow, attention: Mapping[str, Any] | None) -> CatalogEntry:
    """Build one row's :class:`CatalogEntry` from the attention state beside it.

    THE ONE CONSTRUCTION SITE, and that is the point of the function. Every
    feature of the entry above ``status_code`` — the precedence itself, the
    ``shows_completion_mark`` predicate, the reason sentences, the ranking — is
    derived state, and a second place that assembled an entry from the same two
    inputs would be free to derive it differently while every existing test
    stayed green. So the list (``load_catalog``) and the desktop feed (which
    publishes the derived pair as a frame) both come through here.

    ``attention`` is ONE session's state as ``AttentionStore.state_many``
    returns it (``unseen``/``kind``/``completion_token``/``anchor_id``/
    ``reason``), or ``None``/``{}`` for a session with no state yet. Read
    defensively by key: an absent store contributes the empty state, which is
    exactly what ``state_many`` hands back for an unknown conversation.
    """
    state = attention or {}
    return CatalogEntry(
        row,
        bool(state.get("unseen", False)),
        str(state.get("kind") or ""),
        str(state.get("completion_token") or ""),
        str(state.get("anchor_id") or ""),
        str(state.get("reason") or ""),
    )


def status_dedupe_key(row: SessionRow, attention: Mapping[str, Any] | None) -> tuple[str, str]:
    """``status_of``'s pair with the CLOCK term removed — the edge channel's key.

    WHY THE PAIR IS NOT ENOUGH (review round 1, MINOR 2). One label carries a
    live clock: the ``wedged`` arm embeds ``format_duration(heartbeat_age_s)``
    (``46s`` -> ``47s`` -> ``1m``), so a wedged session's pair changes with the
    clock alone, with no write and no event behind it — roughly one change per
    second for the 45-59 s window after the beat crosses
    ``HEARTBEAT_TIMEOUT_S``, then one a minute, for every wedged session. A
    channel whose whole promise is "a frame per EVENT" cannot treat that as an
    edge, so it dedupes on this key and still PUBLISHES the pair.

    The key is the same derivation on the same row with the age cleared, which
    is a no-op for every arm but ``wedged`` (that arm is the only reader of
    ``heartbeat_age_s``). So it rides :func:`status_of` rather than restating the
    precedence, and the dedupe cannot drift from what the row says.

    The cost of the rule, stated because it is real: a client that keeps a
    wedged row's frame therefore keeps the label it was published with, and the
    age in that sentence stops advancing until the row's next real edge (or the
    client's own 30 s safety poll) refreshes it. A tooltip's age is not worth a
    frame a second per wedged session, which is the same trade the 15 s
    heartbeat rewrite already makes.
    """
    if row.heartbeat_age_s is None:
        return status_of(row, attention)
    return status_of(row._replace(heartbeat_age_s=None), attention)


def active_of(row: SessionRow, attention: Mapping[str, Any] | None) -> bool:
    """``CatalogEntry.active`` for one row — which SECTION the sidebar files it in.

    A second CALLER of the same home, for the same reason :func:`status_of` is
    one: section membership is derived state (``pending or unseen or
    live_state``), and a caller that restated it would be free to move a row the
    list does not move. The desktop feed asks this beside ``status_of`` because a
    row can need to change section while its pair changes too — a background
    session that finishes goes from "Previous chats" to "Active chats", and
    placement is carried by a LIST read, so the feed owes its client an
    invalidation when that happens (finding 8).
    """
    return entry_for(row, attention).active


def status_of(row: SessionRow, attention: Mapping[str, Any] | None) -> tuple[str, str]:
    """``(status_code, status)`` for one row — the transport spelling and the label.

    A second CALLER of the precedence, never a second home: it builds the same
    :class:`CatalogEntry` :func:`entry_for` builds — the same one
    ``load_catalog`` builds — and returns the same two properties from it. The
    desktop feed publishes this pair as a ``session_status`` frame; the list
    ships it on every row. If the two ever disagree, one of them is not calling
    this function.

    Present because the feed has a ``SessionRow`` and an attention state and
    nothing else: it must not read ``CatalogEntry``, re-order the branches, or
    name a code itself. See :func:`entry_for`.
    """
    entry = entry_for(row, attention)
    return entry.status_code, entry.status


def rank_entries(entries: Sequence[CatalogEntry]) -> tuple[CatalogEntry, ...]:
    """Stable identities survive refreshes, including deterministic recency ties."""
    return tuple(sorted(entries, key=lambda entry: entry.rank))


def session_directory_name(session_id: str) -> bool:
    """Discovery metadata cannot redirect a catalog read outside sessions/."""
    return (
        isinstance(session_id, str)
        and bool(session_id)
        and session_id not in {".", ".."}
        and not any(
            character in "/\\\\" or ord(character) < 32 or ord(character) == 127
            for character in session_id
        )
    )


#: The live-decoration reads whose failure leaves a row's defaults UNKNOWN
#: rather than FALSE, and the one spelling each of them rides the wire under.
#:
#: Gathered here, and sent as data rather than as three booleans, because three
#: readers have to agree on the words: ``SessionRow.degraded`` carries them, the
#: desktop list lifts them onto its response, and a renderer checks them to
#: decide whether it may say "nothing is running". A renderer that spelled its
#: own constants would silently never match, and the failure mode of that is the
#: one this change exists to remove.
#:
#: Adding a fourth source is one word here plus the ``degraded += (...)`` at the
#: read that can fail; nothing else changes shape.
DECORATION_LIVENESS = "liveness"
DECORATION_WAKES = "wakes"
DECORATION_ATTENTION = "attention"
DECORATION_SOURCES = (DECORATION_LIVENESS, DECORATION_WAKES, DECORATION_ATTENTION)


def decorate_rows(
    directory: Path, rows: list[SessionRow], *, include_live: bool = False
) -> list[SessionRow]:
    """Fill in each row's runtime state, and float the ones needing a person.

    Two reads for the whole list: the discovery records say which sessions
    are running, working, attached or not answering, and the wake index says
    which have reminders armed. Best-effort — a picker that cannot read either
    one still lists every session exactly as it did before, because the
    fields are defaulted and the markers simply do not appear.

    BEST-EFFORT IS NOT SILENT, and this is the correction. The fields above are
    DEFAULTS, and a defaulted ``live_state=""`` is indistinguishable from a
    measured "this session is cold" — so when ``registry.scan`` was swallowed,
    ``CatalogEntry.active`` came out ``False`` for every row and the wire said
    ``active: false`` about a store with a running turn in it. The sidebar
    renders exactly that as the expanded "Active chats" section reading
    "Nothing running right now" while the collapsed section holds the rest: a
    swallowed read failure presented to the operator as "all my active chats
    disappeared".

    So a failed read now says so: the affected source is named on every row's
    :attr:`SessionRow.degraded`, the incident is logged at WARNING (``debug`` is
    invisible at the default level, which is why an incident could not be
    reconstructed from the logs afterwards), and the defaults keep their old
    values so nothing that renders today changes shape. A client that ignores
    the new field renders exactly as before; a client that reads it can say
    "I could not tell" instead of asserting a negative it does not know.

    THE STATE IS TAKEN, NOT DERIVED. ``registry.scan`` owns the vocabulary
    (its ``classify`` is the one place ``live``/``wedged``/``stale`` is
    decided) and the mapping below is total over those three words and nothing
    else. The heartbeat AGE comes from that same owner rather than from a
    second subtraction here, so the tooltip cannot disagree with the verdict
    about what a future-dated stamp means; the zombie probe is skipped because
    the verdict has already been reached and a second ``ps`` fork per row per
    poll would buy nothing.
    """
    from local_operator.session.runtime import registry

    #: This poll's failed sources, as a tuple so each row can carry the verdict
    #: by reference rather than being rebuilt per row.
    degraded: tuple[str, ...] = ()
    try:
        scanned = registry.scan(directory)
    except Exception:  # noqa: BLE001 — markers are an enhancement, never a gate
        logger.warning("session catalogue could not read the live records", exc_info=True)
        scanned = []
        degraded += (DECORATION_LIVENESS,)
    try:
        from local_operator.wakes.store import read_index

        wake_index = read_index(directory)
    except Exception:  # noqa: BLE001
        logger.warning("session catalogue could not read the wake index", exc_info=True)
        wake_index = {}
        degraded += (DECORATION_WAKES,)

    live: dict[str, tuple[Any, str]] = {}
    for record, state in scanned:
        session_id = getattr(record, "session_id", "")
        # A ``stale`` VERDICT IS NO RECORD (review round 1, MINOR 1). The pid is
        # gone, so nothing the record says about work in progress is true any
        # more — which is the rule the desktop feed already applies
        # (``DesktopFeed._row_for``) and the reason a dead record's row must not
        # read as busy/attached. Without this the two surfaces disagreed for
        # exactly the poll that reaps the record, and they disagreed on the
        # feed's OWN verdict: the list painted the corpse's ``busy`` while the
        # frame said ``complete``, and because both writers read the same
        # revision counter the client's strictly-greater guard kept the list's
        # wrong value until the next 30 s poll. Taking the rule at BOTH readers
        # removes the divergence rather than documenting it, and the sweep this
        # function's own ``scan`` performs is unaffected: the record is still
        # moved aside, and the row simply stops describing it.
        if session_id and state != "stale":
            live[session_id] = (record, state)

    if include_live:
        from local_operator.resume import is_user_session

        known = {row.id for row in rows}
        rows = list(rows)
        for session_id, (record, _state) in live.items():
            if not session_directory_name(session_id):
                continue
            session_dir = directory / "sessions" / session_id
            if session_id not in known and session_dir.is_dir() and is_user_session(session_dir):
                rows.append(
                    SessionRow(
                        session_id,
                        float(getattr(record, "started_at", 0.0) or 0.0),
                        str(getattr(record, "conversation_name", "") or "Untitled conversation"),
                        created_at=session_created_at(session_dir),
                        degraded=degraded,
                    )
                )
    updated: list[SessionRow] = []
    for row in rows:
        record_state = live.get(row.id)
        live_state = ""
        pending: str | None = None
        leaving = ""
        kind = ""
        if record_state is not None:
            record, state = record_state
            # ``wedged`` here means the owner has stopped reporting, which is a
            # fact about its RECORD and not a diagnosis of its process: a long
            # turn on an in-process runtime produces it while the session is
            # working. The tooltip says exactly that, with the age beside it.
            if state == "wedged":
                live_state = "wedged"
            elif getattr(record, "busy", False):
                live_state = "busy"
            elif not getattr(record, "detached", False):
                live_state = "attached"
            else:
                live_state = "idle"
            pending = getattr(record, "pending", None) or None
            # Only a LIVE record has a kind. A cold row keeps "" so the picker
            # says nothing rather than claiming a session is still an exec run
            # after the process that made it that has gone.
            kind = str(getattr(record, "kind", "") or "")
            # THE DRAIN, carried as the record's own phrase rather than folded
            # into ``live_state``. A signalled runtime IS busy, so the token is
            # not wrong — it is just not the fact a reader needs, and the token
            # is what consumers branch on (``status_code`` is a transport
            # spelling, ``session_category`` ranks on it). A third value there
            # would be a contract change to say something the row can say in a
            # field of its own, which is the same call the CLI's LEAVING COLUMN
            # made instead of adding a token to STATE (design round 2, D3).
            # Defaulted through getattr like the neighbouring live fields: a
            # record written by an OLDER runtime has no such field, and this
            # runs on the poll loop behind ``/resume``.
            leaving = str(getattr(record, "leaving", "") or "")
        entry = wake_index.get(row.id) or {}
        schedules = entry.get("schedules") or () if isinstance(entry, dict) else ()
        age: float | None = None
        if record_state is not None:
            age = registry.classify(record_state[0], check_zombie=False).heartbeat_age_s
        updated.append(
            row._replace(
                live_state=live_state,
                pending=pending,
                leaving=leaving,
                wakes=len(schedules),
                wakes_dormant=bool(isinstance(entry, dict) and entry.get("stopped_at")),
                kind=kind,
                heartbeat_age_s=age,
                # THIS poll's verdict, not an accumulation: a value inherited
                # from an earlier decoration of the same row would outlive the
                # failure it described.
                degraded=degraded,
            )
        )
    return sorted(updated, key=lambda row: 0 if row.pending else 1)


#: Rows the poll materialises. The sidebar paints a fixed window (~38 rows at a
#: usual height) and pages within what it holds, so the untruncated answer
#: `/resume` wants is waste here: on a 665-directory store the poll built 56
#: rows every 2 s and spent 92-99% of itself doing it. Headroom well past the
#: viewport keeps paging and ranking honest without materialising the tail.
CATALOG_SCAN_LIMIT = 200

#: Newest subagent runs the sidebar's ⌥ layer lists when it is switched on.
#: One screenful of headroom past the ~38-row window, matching the reasoning
#: behind CATALOG_SCAN_LIMIT. No paging: the layer answers "what just ran".
#:
#: WHAT THIS CAP DOES AND DOES NOT BOUND. The cap bounds the PAGE, not the
#: selection. The per-row reads on the capped page — `origin.json` and
#: `session_created_at` — are O(cap): measured flat at 85 `origin.json` reads
#: with the hidden population at 50, 100, 400 and 800. What is NOT capped is
#: the step-2 selection sweep in `load_catalog`, which stats every hidden
#: directory to find the newest CAP of them: that is O(hidden population),
#: linear at a stable ~6.0 µs/dir measured from 250 to 4000 dirs, and it is
#: the term that scales.
#:
#: So the ON-path delta grows with the store, not with this constant. The
#: +6.6 ms measured today was ruled ACCEPTED-with-documentation rather than a
#: regression because the original +2.7 ms budget was set on a 462-hidden
#: store and the store outgrew it — the sweep is the prescribed algorithm
#: costing what it costs at a larger population, not a leak of the capped
#: reads into the population. Memoizing the sweep is the follow-up, and it
#: becomes worth its own invalidation surface at roughly 2,000 hidden
#: directories (~12 ms, 0.6% of the 2 s poll), not before.
#:
#: The ruling, with the measurements: `BRIEF-sidebar-slice-b.md`,
#: "## Lead check — round 1".
SUBAGENT_LAYER_CAP = 40

#: `session_id -> ((activity_mtime, transcript_size), SessionRow)`. A row's
#: name and fork mark change only when its transcript does, and the scan
#: already stats that file to rank the session, so the key is free. Only the
#: DURABLE fields are cached: `live_state`, `pending`, `wakes` and `unseen` are
#: layered on afterwards by `decorate_rows`/attention on every poll, because
#: caching a live fact would freeze the list.
_ROW_CACHE: dict[Path, tuple[tuple[float, int], SessionRow]] = {}


def _row_stat_key(session_dir: Path) -> tuple[float, int] | None:
    """``(activity_mtime, size)`` for the transcript, or ``None`` if unreadable.

    Deliberately the same file :func:`session.retention.session_activity`
    ranks by, so a row whose key is unchanged is a row whose transcript has not
    been appended to — which is exactly the condition under which its name and
    fork mark cannot have changed. Size is carried alongside mtime because a
    coarse filesystem timestamp can hide an append inside the same second.
    """
    from local_operator.session.retention import TRANSCRIPT_FILENAME

    try:
        stat = (session_dir / TRANSCRIPT_FILENAME).stat()
    except OSError:
        return None
    return (stat.st_mtime, stat.st_size)


def cached_session_rows(
    directory: Path,
    limit: int = CATALOG_SCAN_LIMIT,
    *,
    candidates: list[tuple[str, float, str]] | None = None,
) -> list[SessionRow]:
    """:func:`recent_session_rows` for the poll, memoized on transcript stat.

    The ``O(directories)`` scan underneath is NOT what this avoids — it still
    runs, and bounding it is what ``limit`` does. What this avoids is the
    per-row work above the scan: the bounded head read that builds the name,
    and the fork-title probe, on rows whose transcript has not been appended to
    since the last poll two seconds ago.

    Deliberately reimplements ``recent_session_rows``'s loop instead of calling
    it, because the saving is inside that loop; the scan it calls is the shared
    one, so ranking and visibility stay identical to ``/resume``. Rows absent
    from the current answer are dropped, keeping the cache bounded by the live
    store rather than by every session ever listed.
    """
    from local_operator.resume import (
        ORIGIN_FORK,
        _recent_sessions_with_origin,
        session_name,
        wears_inherited_title,
    )

    rows: list[SessionRow] = []
    fresh: dict[Path, tuple[tuple[float, int], SessionRow]] = {}
    selected = (
        candidates if candidates is not None else _recent_sessions_with_origin(directory, limit)
    )
    # Resolved ONCE for the store, not per row. ``Path.resolve()`` is a
    # ``realpath`` — an ``lstat`` per path component — and the loop below called
    # it twice for every row, which measured 2,880 realpath calls and 1,440
    # lstats per poll on a 144-row listing: 21% of the poll's profile spent
    # building a dictionary key. The variable part of that path is the session
    # id, which is a single directory name, so resolving the parent captures
    # everything a symlinked store could redirect.
    #
    # Safe even for the exotic case of an individual session directory being
    # itself a symlink: this key is only ever an in-process memo key, never an
    # I/O path, so a key that differs from the fully-resolved one can only cost
    # a cache MISS (a row rebuilt from disk, which is correct by construction),
    # never a wrong or stale row.
    root = (directory / "sessions").resolve()
    for session_id, mtime, origin in selected:
        session_dir = directory / "sessions" / session_id
        key = _row_stat_key(session_dir)
        cached = _ROW_CACHE.get(root / session_id)
        if key is not None and cached is not None and cached[0] == key:
            # Same transcript bytes as last poll: the name and the fork mark
            # cannot have changed, so neither read is repeated. `mtime` is
            # taken fresh from the scan regardless — it also tracks the inbox
            # spool. It updates displayed age, never immutable creation order.
            row = cached[1]._replace(mtime=mtime)
        else:
            row = SessionRow(
                session_id,
                mtime,
                session_name(session_dir),
                forked=origin == ORIGIN_FORK and wears_inherited_title(session_dir),
                created_at=session_created_at(session_dir),
            )
        rows.append(row)
        if key is not None:
            fresh[root / session_id] = (key, row)
    _ROW_CACHE.clear()
    _ROW_CACHE.update(fresh)
    return rows


def _subagent_marker(session_dir: Path) -> tuple[str, str]:
    """``(agent, label)`` from a subagent's ``origin.json``; ``("", "")`` if unreadable.

    Best-effort by contract, like every other marker read on this path: a
    missing, truncated or non-object marker costs the row its role and task
    label, never the row itself.
    """
    try:
        payload = json.loads((session_dir / "origin.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ("", "")
    if not isinstance(payload, dict):
        return ("", "")
    values = []
    for key in ("agent", "label"):
        value = payload.get(key)
        values.append(value if isinstance(value, str) and value else "")
    return (values[0], values[1])


def subagent_population(directory: Path) -> int:
    """How many hidden subagent runs the store holds — the footer chip's count.

    A full second scan of the store: ``_scan_sessions`` is not memoized, so
    this costs about as much as ``load_catalog`` itself (+2.36 ms measured on a
    156-visible/462-hidden store). Call it on a slow cadence — the sidebar
    reads it on open and then every fifteenth poll — never once per poll.
    """
    from local_operator.resume import _scan_sessions

    return len(_scan_sessions(directory)[1])


def load_catalog(
    directory: Path,
    limit: int = CATALOG_SCAN_LIMIT,
    *,
    include_subagents: bool = False,
    pinned_hidden_ids: Sequence[str] = (),
    pinned_off_page: Sequence[str] = (),
) -> list[CatalogEntry]:
    """Rank a shared lightweight candidate snapshot before materializing a page.

    Discovery already stats the whole namespace. Applying a recency cap before
    attention lost old unread work; reading names for the entire store would
    undo the sidebar's bounded I/O. Rank cheap rows first, then hydrate only the
    requested prefix through the existing transcript-stat cache.

    STRICT ABOUT THE STORE, TOLERANT ABOUT THE DECORATION, and the difference is
    deliberate. This is the listing a UI ADOPTS AS MEMBERSHIP — the desktop
    sidebar replaces the rows it is showing with this answer, and the TUI's sets
    its entries from it — so a store that exists but cannot be walked raises
    (:class:`SessionStoreUnavailable`, which the desktop route answers as a
    retryable 503) rather than being reported as "you have no conversations".
    Decorations are the opposite case: they never change WHICH rows are
    returned, only what is claimed about them, so a read that fails here is
    named on each row's ``degraded`` and the listing still stands.

    ``include_subagents`` adds a capped page of the hidden subagent population
    as a SEPARATE layer, and ``pinned_hidden_ids`` keeps individually pinned
    hidden sessions resolvable while that layer is off. Both are keyword-only
    and default to the behaviour every existing caller already has: with the
    layer off this function issues exactly the syscalls it did before.

    ``pinned_off_page`` keeps individually pinned sessions resolvable when the
    PAGE does not carry them — the same promise ``pinned_hidden_ids`` makes on
    the other axis, and a different one: a hidden id is absent because the
    catalogue never built it, while an off-page id IS built here and is dropped
    by the ``limit`` slice below. A caller that renders a pinned section must
    have both, because a pin is a durable statement by the user and a recency
    window is a property of the listing. The extras come back APPENDED, after
    the page and in the ranking's own order (every extra ranks below every page
    row by construction, so the concatenation IS rank order).
    """
    from dataclasses import replace

    from local_operator.resume import (
        _scan_sessions,
        _scanned_entries,
        _store_error_detail,
    )
    from local_operator.session.attention import AttentionStore, conversation_identity
    from local_operator.session.errors import SessionStoreUnavailable
    from local_operator.session.retention import (
        DESKTOP_MARKER_NAME,
        TRANSCRIPT_FILENAME,
    )

    # The scan's second return value is every directory it established is not
    # the user's own session. Taken here rather than recomputed because it is
    # what removes this function's own O(store) stat; see the desktop-probe loop
    # below for why a hidden directory cannot carry a desktop marker.
    #
    # ``strict=True`` is this function's own declaration, not a global policy:
    # see the docstring above, and ``_scan_sessions`` for the boundary it draws
    # between a store that is not there (an empty listing, still) and a store
    # that cannot be read (an unavailable one).
    candidates, hidden = _scan_sessions(directory, strict=True)
    source = {session_id: (session_id, mtime, origin) for session_id, mtime, origin in candidates}
    # -- the opt-in subagent layer (PROPOSAL 5a) ----------------------------
    #
    # Every syscall this layer costs is inside this branch: with both keyword
    # arguments at their defaults nothing below runs, and the poll is byte-for-
    # byte the scan it was. `hidden` is already in hand from the scan above, so
    # the OFF path does not even pay a directory read to learn it is off.
    #
    # THESE ROWS DELIBERATELY BYPASS `decorate_rows` AND THE ATTENTION LOOKUP
    # BELOW. `CatalogEntry.active` is `pending or unseen or live_state`, and the
    # attention store keys on conversation identity -- which answers for
    # subagent ids too, 45% of them carrying an unseen receipt. A sub row placed
    # in `rows` therefore comes out `active=True` and sections into Active
    # Sessions, above the user's own work. No filter inside that path avoids it;
    # the only fix is not to be on the path. So these are built by hand, in
    # their own list, with the quiet fields left at their empty defaults, and
    # they rejoin the main flow at exactly one point: the `rank_entries` call.
    subagent_entries: list[CatalogEntry] = []
    if include_subagents or pinned_hidden_ids:
        pinned = [session_id for session_id in pinned_hidden_ids if session_id in hidden]
        wanted = set(hidden) if include_subagents else set()
        wanted.update(pinned)
        stamped: list[tuple[float, str]] = []
        for session_id in wanted:
            transcript = directory / "sessions" / session_id / TRANSCRIPT_FILENAME
            try:
                stamped.append((transcript.stat().st_mtime, session_id))
            except OSError:
                # No readable transcript: nothing to hydrate a row from, and
                # `_row_stat_key` would decline to cache it anyway.
                continue
        stamped.sort(key=lambda item: (-item[0], item[1]))
        mtimes = {session_id: mtime for mtime, session_id in stamped}
        # Pinned ids are exempt from the cap -- a pin to the 300th-oldest run
        # must still resolve, or the pin renders as nothing at all.
        selected = list(
            dict.fromkeys(
                [session_id for _, session_id in stamped[:SUBAGENT_LAYER_CAP]]
                + [session_id for session_id in pinned if session_id in mtimes]
            )
        )
        for session_id in selected:
            session_dir = directory / "sessions" / session_id
            agent, label = _subagent_marker(session_dir)
            # Added to `source` ONLY -- never to `candidates`, never to `rows`.
            # `source` is what the single `cached_session_rows` call below looks
            # rows up in, so this alone is what buys these rows their real name.
            source[session_id] = (session_id, mtimes[session_id], "subagent")
            subagent_entries.append(
                CatalogEntry(
                    SessionRow(
                        session_id,
                        mtimes[session_id],
                        "",
                        # Stamped, never left at the 0.0 default: rows that all
                        # tie there fall through to the id tie-break, which
                        # silently reverses newest-first order.
                        created_at=session_created_at(session_dir),
                    ),
                    subagent=True,
                    agent=agent,
                    label=label,
                )
            )
    # Creation time is the immutable ordering key (#800), so every construction
    # site must stamp it. Rows left at the 0.0 default all tie and fall through
    # to the session-id tie-break, which silently reverses newest-first order.
    rows = [
        SessionRow(
            session_id,
            mtime,
            "",
            created_at=session_created_at(directory / "sessions" / session_id),
        )
        for session_id, mtime, _ in candidates
    ]
    # One directory read plus a stat per unlisted candidate, NOT
    # ``glob("*/desktop.json")``. The glob looks equivalent and is not: a
    # pattern whose wildcard is a DIRECTORY component makes pathlib open and
    # enumerate every session directory to learn one filename, so this line
    # alone issued 1,946 ``scandir`` calls per poll on the reporting machine
    # — 1.20 s of a 6.47 s profile, the single largest term in it — to find
    # the ONE marker that store actually contains. Asking about the file
    # directly answers the same question with a stat.
    #
    # Cheaper still, the ``in source`` test now runs BEFORE the filesystem is
    # touched rather than after the glob has already paid for the directory:
    # every session that is already listed costs nothing here at all.
    try:
        entries = os.scandir(directory / "sessions")
    except FileNotFoundError:
        # No store to probe, which the scan above has already answered as an
        # empty listing. NOT a failure, for the same reason a failed OPEN is
        # not: a fresh install must list its (zero) conversations, not 503.
        entries = None
    except OSError as error:
        # Same boundary as the scan's own open, and a sharper reason to surface
        # it than "the store is unreadable": the rows this loop contributes are
        # the desktop-created sessions that have no transcript YET — the newest
        # thing in the store, and the row a person is most likely to be looking
        # for. Answering "those rows do not exist" for a directory that could
        # not be read is the defect this whole change is about, one layer down.
        #
        # The per-entry ``except OSError`` INSIDE the loop is a different
        # question and keeps its answer: "no desktop.json here" is the common
        # expected reply to that stat, not a broken read.
        logger.warning("session catalogue could not probe for desktop records", exc_info=True)
        raise SessionStoreUnavailable(_store_error_detail(error)) from error
    if entries is not None:
        with entries:
            for entry in _scanned_entries(entries):
                if entry.name in source:
                    continue
                # A directory the scan established is a subagent/hidden session
                # cannot carry a desktop marker, so the stat below asks a
                # question whose answer is already known. ``desktop.json`` has
                # exactly ONE WRITER FUNCTION — ``write_desktop_marker`` in
                # ``server/utils/desktop_sessions.py`` — reached by two CALLERS:
                # ``DesktopSessions.create``, which mints a fresh ``uuid4``
                # directory, and the move route, which rewrites the marker of an
                # EXISTING user session. Neither ever adds a marker to a
                # directory that is hidden or a child of one, which is the
                # property this skip relies on: a move can only touch a
                # directory the catalogue already shows as a session.
                # Skipping these is HALF the saving of the inode-qualified scan,
                # because the hidden population is ~91% of the store and every
                # one of them landed here.
                if entry.name in hidden:
                    continue
                try:
                    # No ``is_dir()`` guard: a successful stat of a file INSIDE
                    # the entry already proves it is a directory, so asking
                    # first would spend a stat per entry to learn what this one
                    # tells us for free. The common answer here is ENOENT.
                    marker_mtime = os.stat(os.path.join(entry.path, DESKTOP_MARKER_NAME)).st_mtime
                    # A marker is a draft fallback, never a competing source of
                    # historical title/mtime for a transcript that fell beyond a
                    # previous page bound.
                    if os.path.exists(os.path.join(entry.path, TRANSCRIPT_FILENAME)):
                        continue
                except OSError:
                    # Includes the common case: no ``desktop.json`` here, which
                    # the glob expressed as a non-match rather than an error.
                    continue
                # Checked only for an entry that actually carries a marker.
                # It is a containment guard on names that may come from
                # discovery metadata, not a filter this loop needs per entry,
                # and it is pure-Python per-character work worth keeping off
                # the path taken by every directory in the store.
                if not session_directory_name(entry.name):
                    continue
                rows.append(
                    SessionRow(
                        entry.name,
                        marker_mtime,
                        "",
                        created_at=session_created_at(Path(entry.path)),
                    )
                )
    rows = decorate_rows(directory, rows, include_live=True)
    identities = {row.id: conversation_identity(directory / "sessions" / row.id) for row in rows}
    attention: dict[str, dict[str, Any]] = {}
    try:
        attention = AttentionStore(directory / "attention.db").state_many(identities.values())
    except (sqlite3.Error, OSError):
        # The third read whose failure used to be published as a confident
        # negative: the defaults this leaves behind are ``unseen=False`` and an
        # empty completion kind, so an unread completion would render as an
        # ordinary read one — and ``unseen`` is part of ``CatalogEntry.active``,
        # so the row would leave the Active section on the strength of a read
        # that did not happen.
        #
        # Named ON THE ROWS rather than carried alongside them because the
        # catalogue's return type is a list of entries and every consumer walks
        # it: a sibling value would be a second channel every caller has to know
        # about, and a caller that forgot it would be back to the silent
        # negative. Stamped here rather than in ``decorate_rows`` because this
        # is the only read of that store, and it happens after the decoration.
        logger.warning("session catalogue could not read attention state", exc_info=True)
        rows = [row._replace(degraded=row.degraded + (DECORATION_ATTENTION,)) for row in rows]
    ranked = list(
        rank_entries(
            [entry_for(row, attention.get(identities[row.id])) for row in rows]
            # The ONE join point. Sub entries are concatenated here rather than
            # being members of `rows`, so they never pass through the
            # decoration and attention work above -- and this stays a single
            # `rank_entries` call over a single list.
            + subagent_entries
        )
    )
    # THE PAGE, THEN THE PINS THAT FELL OUTSIDE IT. The slice is a RECENCY
    # window, and a pin is the one thing in this listing that is not recency: it
    # is a durable statement the user made about a conversation. On a store
    # larger than the client's page a pinned session is therefore UNRENDERABLE
    # without this — the row the client needs to draw in its pinned section is
    # exactly the row the window removes — so the caller that renders pins names
    # them and gets them back here.
    #
    # WHAT THIS COSTS: nothing measurable, and that is the reason it lives here
    # rather than in a caller. Every candidate's row, decoration and attention
    # lookup has ALREADY happened by this point — the slice is the only thing
    # that discarded these entries — so an extra costs one membership test, not
    # a scan, a hydration or a second `cached_session_rows` call. A caller-side
    # union would have to re-scan the store or re-hydrate by id, and the id
    # lookup it would need is the one thing this function does not return.
    #
    # SILENTLY ABSENT WHEN IT CANNOT BE RESOLVED, deliberately: an id that is
    # not in ``ranked`` at all — a hidden (delegated) run, which never reaches
    # ``candidates``, or a directory that has since been deleted — is simply not
    # appended, rather than raising. The caller asked for a row it would like to
    # render, not for a promise this store cannot keep.
    entries = ranked[:limit]
    if pinned_off_page:
        # The window is the caller's own page bound, so a pinned row sitting
        # exactly AT that bound is carried too — it is a row the caller's page
        # does not carry, which is the whole condition. Order is preserved: the
        # extras rank below every page row by construction, so appending them
        # keeps the result in rank order.
        wanted = set(pinned_off_page)
        entries += [entry for entry in ranked[limit:] if entry.id in wanted]
    # KNOWN LIMITATION, decided rather than missed. This slice applies to the
    # COMBINED list, so a store with more than ~160 visible sessions cannot fit
    # both populations in CATALOG_SCAN_LIMIT. Sub rows do not lose that race:
    # their `session_category` inputs are all falsy by construction, giving them
    # the same tier as a cold visible row, where the tie-break is `-created_at`
    # and subagent runs are recent. So what is pushed past the slice is the
    # user's OLDEST COLD sessions, never an active row. Raising the limit would
    # restore the per-poll row-building cost the comment on CATALOG_SCAN_LIMIT
    # exists to document removing, so it is deliberately not raised.
    #
    # A PINNED ID AMONG THEM IS NO LONGER PUSHED OUT, and that is the one thing
    # `pinned_off_page` changes about this sentence: the row is still sliced off
    # the page, and it comes back appended, which is cheaper than raising the
    # limit for everyone to serve the few rows a user pinned.
    # Pinned by `test_the_layer_competes_for_the_page_on_a_full_store`.
    named = {
        row.id: row
        for row in cached_session_rows(
            directory, candidates=[source[entry.id] for entry in entries if entry.id in source]
        )
    }
    return [
        (
            replace(
                entry,
                row=entry.row._replace(name=named[entry.id].name, forked=named[entry.id].forked),
            )
            if entry.id in named
            else entry
        )
        for entry in entries
    ]
