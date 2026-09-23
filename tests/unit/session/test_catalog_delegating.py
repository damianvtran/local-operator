"""The ``delegating`` state: an idle parent that still owns running children.

WHAT THIS FILE IS FOR. The state was added because a session whose own turn
ended while its children still run read as "no activity" on every surface — the
TUI sidebar, the desktop sidebar and the phone. That is a PRECEDENCE claim as
much as a new word: the state has to appear where it was invisible and nowehere
else, it has to lose to every louder fact about the row, and it must not move a
row. Each of those is a separate way to get it wrong, and each is asserted here
rather than being left to the frame that happened to be captured.

WHY THE TABLE IS EXHAUSTIVE ABOUT COLLISIONS. ``CatalogEntry.status`` is one
ladder with one order, and the rung only exists if the states ABOVE it keep
winning. A test that only checked "idle + 2 children says delegating" would pass
against an implementation that put the new arm at the TOP of the ladder, which
would mask a parked gate, an unread error and an attached session — the three
surfaces' most important rows.
"""

from __future__ import annotations

import time

import pytest

from local_operator.resume import SessionRow
from local_operator.session.catalog import CatalogEntry, order_key_of, status_of

NOW = time.time()


def entry(**row: object) -> CatalogEntry:
    """A catalogue entry for one hand-built row. No disk, no scan."""
    return CatalogEntry(
        SessionRow("x" * 12, NOW, "a conversation", **row)  # type: ignore[arg-type]
    )


class TestTheLabelSpellsTheCount:
    """The count rides INSIDE ``status``, which is what makes an edge publish.

    The status channel's dedupe key is ``(code, label)`` with the clock term
    removed, so the count has to be part of the label for ``0 -> 2`` and
    ``2 -> 1`` to be edges at all. These assertions are therefore about the
    WIRE, not about cosmetics.
    """

    @pytest.mark.parametrize(
        ("running", "queued", "expected"),
        [
            # The complaint's own shape, and the singular at one — the
            # ``Scheduled (1 wake)`` rule, applied to a second count.
            (2, None, "2 subagents running"),
            (1, None, "1 subagent running"),
            # Both answers, when there are two.
            (2, 3, "2 subagents running · 3 queued"),
            (1, 1, "1 subagent running · 1 queued"),
            # Parked children with nothing spending: NOT idle.
            (0, 2, "2 subagents queued"),
            (0, 1, "1 subagent queued"),
            # A reported zero on the other count is noise, not information.
            (2, 0, "2 subagents running"),
        ],
    )
    def test_the_words(self, running: int, queued: int | None, expected: str) -> None:
        got = entry(live_state="idle", subagents_running=running, subagents_queued=queued).status
        assert got == expected

    def test_the_code_is_the_transport_spelling_and_does_not_vary_with_the_count(self) -> None:
        """One code for the state, whatever the number.

        The desktop renderer paints glyph AND ink from the code alone, so the
        code is what carries the state there; the count is what makes the LABEL
        move. If the code varied by count, every count change would be a new
        state code as far as a client's glyph table is concerned.
        """
        for running, queued in ((2, None), (1, None), (2, 3), (0, 2)):
            assert (
                entry(
                    live_state="idle", subagents_running=running, subagents_queued=queued
                ).status_code
                == "delegating"
            )


class TestAnUnreportedCountIsNotZero:
    """``None`` means "this build does not report", which is not "no children".

    ``SessionRecord.subagents_running``/``subagents_queued`` are ``int | None``
    and a record written before the fields existed carries neither. Rendering
    that as ``0`` would assert a measurement nobody made, on the surface whose
    whole job is telling the operator what is happening.
    """

    def test_an_absent_count_never_reaches_the_state(self) -> None:
        assert entry(live_state="idle").status == "Ready"
        assert entry(live_state="idle").status_code == "idle"
        assert entry(live_state="idle", subagents_running=None).status == "Ready"
        assert entry(live_state="idle", subagents_running=0, subagents_queued=0).status == "Ready"

    def test_an_unreported_queued_count_is_never_printed_as_zero(self) -> None:
        """The one arm where the two facts could be confused.

        A record reporting 2 running and no ``queued`` field must say
        "2 subagents running" — not "2 subagents running · 0 queued", which
        would invent a measurement, and not "2 subagents queued", which would
        misplace the children it does know about.
        """
        only_running = entry(live_state="idle", subagents_running=2, subagents_queued=None)
        assert only_running.status == "2 subagents running"
        assert "0" not in only_running.status

    @pytest.mark.parametrize("bad", ["2", 2.5, True, -1, [2]])
    def test_a_corrupt_count_cannot_raise_or_invent_a_state(self, bad: object) -> None:
        """``from_json`` does no type validation, and this is the first arithmetic.

        The poll loop behind ``/resume`` runs this; a ``str`` or a ``list`` must
        not raise there, and a merely-numeric wrong value must not render as
        fact. ``True`` is called out because ``bool`` is an ``int`` subclass and
        would otherwise count as one subagent.
        """
        assert entry(live_state="idle", subagents_running=bad).status == "Ready"
        assert entry(live_state="idle", subagents_running=bad).status_code == "idle"


class TestEveryLouderRungStillWins:
    """U5: the state must not mask a gate, a stop, a live turn, a receipt or a session.

    Each row below is built TWICE — with and without children — and both copies
    must render identically. That is the property: adding the new rung left every
    state above it exactly where it was.
    """

    CASES: list[tuple[str, dict[str, object], dict[str, object], str, str]] = [
        # A parked gate: a person is blocked on this row right now.
        ("approval gate", {"pending": "approval"}, {}, "approval", "Approval needed"),
        ("free-text question", {"pending": "ask"}, {}, "answer", "Answer needed"),
        # The parent's own lane, which is the fact this state is derived from
        # being NOT true: it must not be claimed on top of a live turn.
        ("busy parent", {"live_state": "busy"}, {}, "busy", "Working"),
        (
            "wedged owner",
            {"live_state": "wedged"},
            {},
            "wedged",
            "Not answering · process alive",
        ),
        # The receipt. Two of the three pre-implementation reviews put this
        # first, and `shows_completion_mark` is the one shared arbiter for
        # "mark or live state" — an unread outcome outranks work the operator
        # can still read about by opening the session.
        ("unseen completion", {}, {"unseen": True}, "complete", "Unseen completion"),
        (
            "unseen error",
            {},
            {"unseen": True, "completion_kind": "error"},
            "error",
            "Unseen error",
        ),
        (
            "unseen interruption",
            {},
            {"unseen": True, "completion_kind": "interrupted"},
            "interrupted",
            "Unseen interruption",
        ),
        # "Where am I?" — the row the user is sitting in.
        ("attached", {"live_state": "attached"}, {}, "attached", "Open"),
        # A runtime committed to exiting, which in `status` already outranks
        # `busy`. Gated in the row predicate, so `status_code` cannot publish
        # `delegating` beside a "Leaving…" tooltip.
        ("leaving", {"leaving": "Draining"}, {}, "idle", "Draining"),
    ]

    @pytest.mark.parametrize(
        ("label", "row", "entry_kwargs", "code", "words"),
        CASES,
        ids=[case[0] for case in CASES],
    )
    def test_the_rung_above_is_untouched_by_children(
        self,
        label: str,
        row: dict[str, object],
        entry_kwargs: dict[str, object],
        code: str,
        words: str,
    ) -> None:
        base_row: dict[str, object] = {"live_state": "idle", **row}
        plain = CatalogEntry(
            SessionRow("x" * 12, NOW, "a conversation", **base_row),  # type: ignore[arg-type]
            **entry_kwargs,  # type: ignore[arg-type]
        )
        with_children = CatalogEntry(
            SessionRow(
                "x" * 12,
                NOW,
                "a conversation",
                subagents_running=2,
                subagents_queued=1,
                **base_row,  # type: ignore[arg-type]
            ),
            **entry_kwargs,  # type: ignore[arg-type]
        )
        assert (plain.status_code, plain.status) == (code, words), label
        assert (with_children.status_code, with_children.status) == (code, words), label


class TestTheWakeRungIsBelowIt:
    """An armed wake is a FUTURE fact; children running are happening now.

    ``◷`` outranks ``●`` because bare residency is the least informative thing
    true of a live row. ``⇉`` is not that, so the wake does not outrank it — the
    complaint's own shape is an idle parent with a wake armed and children
    running, which used to say "Scheduled".
    """

    def test_an_armed_wake_loses_to_running_children(self) -> None:
        armed = entry(live_state="idle", wakes=2, subagents_running=2)
        assert armed.status_code == "delegating"
        assert armed.status == "2 subagents running"

    def test_the_wake_rung_is_otherwise_unchanged(self) -> None:
        assert entry(wakes=2).status == "Scheduled (2 wakes)"
        assert entry(wakes=1, wakes_dormant=True).status == "Stopped (1 wake dormant)"
        dormant_live = entry(live_state="idle", wakes=1, wakes_dormant=True)
        assert dormant_live.status == "Ready"


class TestAddingTheStateMovesNoRow:
    """ORDER: a delegating row keeps its tier and its rank.

    ``rank``'s docstring is the record of this exact mistake — a key applied
    inside a tier reordered rows that the glyph ladder deliberately ranks
    against each other. A new state must not be a new ordering input, so this
    asserts the key is byte-identical with and without the counts, and that the
    tier is the ordinary live-session tier.
    """

    def test_rank_and_order_key_are_unchanged(self) -> None:
        from local_operator.session.catalog import session_category

        plain = CatalogEntry(SessionRow("x" * 12, NOW, "a conversation", live_state="idle"))
        delegating = entry(live_state="idle", subagents_running=2, subagents_queued=1)
        assert delegating.rank == plain.rank
        assert order_key_of(delegating.row, None) == order_key_of(plain.row, None)
        assert delegating.rank[0] == session_category(
            pending=False, busy=False, unseen=False, kind="", live=True
        )

    def test_the_row_stays_in_the_active_section(self) -> None:
        assert entry(live_state="idle", subagents_running=2).active is True


class TestTheCountsReachTheRowFromTheRecord:
    """``decorate_rows`` is where the two numbers enter the model.

    The record the decorator already parsed carries them; the defect was that
    they were dropped on the floor. Asserted through the real scan and a
    published record rather than a hand-built row, because the plumbing is the
    part that was missing.

    RECORDS ARE KEYED BY PID (``run/mobile/<pid>.json``), which is why the two
    cases below are two tests: publishing a second record from the same pid
    OVERWRITES the first, so a single fixture cannot hold both a
    counts-bearing record and a pre-field one.
    """

    @staticmethod
    def _publish(root, session_id: str, *, pid: int, **extra: object):
        import os

        from local_operator.session.runtime import registry
        from local_operator.session.runtime.types import SessionRecord

        return registry.publish(
            SessionRecord(
                pid=pid if pid else os.getpid(),
                kind="tui",
                session_id=session_id,
                conversation_name="parent",
                cwd=str(root),
                model_label="p/m",
                control_port=1234,
                control_key="k",
                detached=True,
                **extra,  # type: ignore[arg-type]
            ),
            root,
        )

    def test_a_published_count_lands_on_the_row(self, tmp_path) -> None:
        import os

        from local_operator.resume import SessionRow
        from local_operator.session.catalog import decorate_rows

        (tmp_path / "run" / "mobile").mkdir(parents=True)
        self._publish(tmp_path, "a" * 12, pid=os.getpid(), subagents_running=2, subagents_queued=1)
        rows = [SessionRow(id="a" * 12, mtime=0.0, name="parent")]
        row = decorate_rows(tmp_path, rows)[0]
        assert (row.subagents_running, row.subagents_queued) == (2, 1)
        assert CatalogEntry(row).status == "2 subagents running · 1 queued"

    def test_an_old_builds_record_reports_nothing_rather_than_zero(self, tmp_path) -> None:
        """A record written BEFORE the fields existed, read by this binary.

        Stripped from the on-disk JSON rather than left at the dataclass
        default, because that is the difference this case is about: an absent
        KEY has to survive the read as "unknown" and not become the ``0`` that
        asserts "no children" about a session nobody asked.
        """
        import json

        from local_operator.resume import SessionRow
        from local_operator.session.catalog import decorate_rows

        (tmp_path / "run" / "mobile").mkdir(parents=True)
        old = self._publish(tmp_path, "b" * 12, pid=1, subagents_running=2)
        payload = json.loads(old.read_text())
        del payload["subagents_running"], payload["subagents_queued"]
        old.write_text(json.dumps(payload), encoding="utf-8")
        rows = [
            SessionRow(id="b" * 12, mtime=0.0, name="old build"),
            SessionRow(id="c" * 12, mtime=0.0, name="no record at all"),
        ]
        by_id = {row.id: row for row in decorate_rows(tmp_path, rows)}
        for session_id in ("b" * 12, "c" * 12):
            row = by_id[session_id]
            assert row.subagents_running is None, session_id
            assert row.subagents_queued is None, session_id
            assert CatalogEntry(row).status_code != "delegating", session_id

    def test_status_of_carries_the_same_pair_the_entry_does(self) -> None:
        """The free functions are what the transport calls; they may not diverge."""
        row = SessionRow("x" * 12, NOW, "a conversation", live_state="idle", subagents_running=3)
        assert status_of(row, None) == (CatalogEntry(row).status_code, CatalogEntry(row).status)
        assert status_of(row, None) == ("delegating", "3 subagents running")
