"""The activity head: the order the chats sidebar reads its first page in.

THE DEFECT THIS FILE PINS. A plain conversation, once its completion was read
and its runtime drained, left the sidebar entirely on the operator's store: the
rank order fills the head page with a wake band (the operator's store carries
~44 scheduled rows) and the state rows above it, so the conversation the reader
had JUST USED sorted past ``limit`` and was never fetched — measured rank 61 of
62 on a rig store carrying 60 wake rows, and ranks 74-95 for the operator's own
recent conversations. The sidebar's own contract says the opposite: its
pagination is "show the latest 10", its sections partition by the ACTIVITY
clock, and it re-orders whatever it holds by that same clock. ``sort=activity``
is that membership rule, at the source: rows that need the reader first
(``CatalogEntry.active``), then everything else newest-activity-first.

WHAT THIS FILE IS CAREFUL ABOUT. The rank order is NOT changed — the TUI
picker's wake band asked for it and keeps it — so the first tests assert the
default page is the rank prefix, byte-for-byte. The activity order is a second
KEY on the same entry, chosen per request and recorded in the cursor, so the
cursor tests assert both directions of the refusal (a position from the other
order answers a first page) and the walk asserts exhaustive, repeat-free
continuation.

The helpers mirror ``test_catalog_scope``: ``created`` is stamped through
``created_at.json`` (``st_birthtime`` is macOS-only, and an order asserted from
the fallback would pass here and collapse to the id tie-break in CI) and
``stamp`` is the transcript's activity, the clock the activity order reads.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import pytest

from local_operator.session import catalog
from local_operator.session.catalog import (
    CATALOGUE_SORT_ACTIVITY,
    catalogue_page,
    decode_cursor,
    load_catalog,
)
from local_operator.session.creation import ensure_session_created_at
from local_operator.wakes import store as wakes_store

RANK_HEAD = ""


def _session(
    root: Path,
    session_id: str,
    *,
    created: float = 1_000.0,
    stamp: float | None = None,
    wake: bool = False,
) -> Path:
    """One visible conversation; ``stamp`` is its activity clock.

    ``wake`` writes a real wake index entry through the store's own writer, so
    the band the activity order must not inherit is exercised for real rather
    than assumed from a flag.
    """
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    ensure_session_created_at(directory, created)
    transcript = directory / "transcript.jsonl"
    transcript.write_text("{}\n", encoding="utf-8")
    clock = stamp if stamp is not None else created
    os.utime(transcript, (clock, clock))
    if wake:
        wakes_store.write_entry(
            root,
            session_id,
            cwd=str(root),
            schedules=[
                {
                    "kind": "wake",
                    "id": f"wake-{session_id}",
                    "next_due_at": int((created + 30 * 24 * 3600) * 1000),
                    "prompt": "Resume.",
                }
            ],
        )
    return directory


def _live(root: Path, session_id: str, *, name: str = "live session") -> None:
    """Publish a REAL runtime record for ``session_id`` (this test's own pid).

    Through ``registry.publish`` and a real ``SessionRecord``, because
    ``active`` is read from the decorated row and a hand-built one could agree
    with a wrong reader. ``os.getpid()`` is by construction alive and
    ``publish`` restamps the heartbeat, so the record classifies as live.
    """
    from local_operator.session.runtime import registry
    from local_operator.session.runtime.types import SessionRecord

    registry.publish(
        SessionRecord(
            pid=os.getpid(),
            kind="daemon",
            session_id=session_id,
            conversation_name=name,
            cwd=str(root),
            model_label="p/m",
            control_port=1234,
            control_key="k",
            detached=False,
        ),
        root,
    )


@pytest.fixture(autouse=True)
def _fresh_memos() -> Any:
    """No test inherits another's memos: the store differs every time."""
    catalog._BIRTH_MEMO.clear()
    yield
    catalog._BIRTH_MEMO.clear()


class TestTheDefaultOrderIsUntouched:
    """``sort=""`` answers the listing every caller read before this existed."""

    def test_the_default_page_is_the_rank_prefix(self, tmp_path: Path) -> None:
        for index in range(12):
            _session(tmp_path, f"plain{index:07d}", created=1_000.0 + index)
        for index in range(3):
            _session(tmp_path, f"waked{index:07d}", created=900.0 + index, wake=True)

        ranked = [entry.id for entry in load_catalog(tmp_path, limit=200)]
        page = catalogue_page(tmp_path, limit=5)

        assert [entry.id for entry in page.entries] == ranked[:5]
        # And the wake band still leads the cold rows: the TUI picker's
        # request that this change deliberately leaves alone. Within the band
        # the order is the rank's own (-created_at), so the newest
        # wake-bearing row leads.
        assert ranked[:3] == [f"waked{index:07d}" for index in (2, 1, 0)]

    def test_the_rank_cursor_shape_is_still_version_one(self, tmp_path: Path) -> None:
        _session(tmp_path, "plain0000001", created=1_000.0)
        _session(tmp_path, "plain0000002", created=1_001.0)
        page = catalogue_page(tmp_path, limit=1)

        assert page.next_cursor is not None
        decoded = decode_cursor(page.next_cursor)
        assert decoded is not None
        assert decoded.order == catalog.CURSOR_ORDER_RANK
        assert len(decoded.key) == 4


class TestTheActivityHead:
    """``sort=activity``: active rows first, then newest-activity-first."""

    def test_active_rows_lead_even_with_the_oldest_activity(self, tmp_path: Path) -> None:
        """Membership is not recency: a row that NEEDS the reader always fits.

        The live row is stamped OLDEST of the three and born oldest of the
        three, so a recency-only order and this order give different answers
        and the test can tell which one it got.
        """
        _session(tmp_path, "cold00000001", created=1_100.0, stamp=1_200.0)
        _session(tmp_path, "cold00000002", created=1_200.0, stamp=1_150.0)
        _session(tmp_path, "live00000001", created=1_000.0, stamp=1_000.0)
        _live(tmp_path, "live00000001")

        page = catalogue_page(tmp_path, limit=9, sort=CATALOGUE_SORT_ACTIVITY)

        assert [entry.id for entry in page.entries] == [
            "live00000001",
            "cold00000001",
            "cold00000002",
        ]

    def test_cold_rows_follow_the_activity_clock_not_birth_or_the_wake_band(
        self, tmp_path: Path
    ) -> None:
        """The three cold orderings disagree by construction.

        Birth order is 1,2,3; activity order is 3,1,2; the wake band would put
        row 2 first. Only the activity clock produces ``3,1,2``.
        """
        _session(tmp_path, "cold00000001", created=1_000.0, stamp=1_100.0)
        _session(tmp_path, "cold00000002", created=1_001.0, stamp=1_050.0, wake=True)
        _session(tmp_path, "cold00000003", created=1_002.0, stamp=1_200.0)

        page = catalogue_page(tmp_path, limit=9, sort=CATALOGUE_SORT_ACTIVITY)

        assert [entry.id for entry in page.entries] == [
            "cold00000003",
            "cold00000001",
            "cold00000002",
        ]

    def test_equal_activity_breaks_on_the_id(self, tmp_path: Path) -> None:
        _session(tmp_path, "tie000000002", created=1_000.0, stamp=1_500.0)
        _session(tmp_path, "tie000000001", created=1_100.0, stamp=1_500.0)

        page = catalogue_page(tmp_path, limit=9, sort=CATALOGUE_SORT_ACTIVITY)

        assert [entry.id for entry in page.entries] == ["tie000000001", "tie000000002"]

    def test_the_wake_band_does_not_hold_the_page_against_a_just_used_row(
        self, tmp_path: Path
    ) -> None:
        """THE REGRESSION, in miniature: the head page must reach the new row.

        Thirty wake-bearing rows, all stamped older than the subject, plus the
        subject itself as the newest activity. The rank order fills the head
        page with the wake band and never reaches the subject; the activity
        order must reach it FIRST.
        """
        for index in range(30):
            _session(
                tmp_path,
                f"waked{index:07d}",
                created=900.0 + index,
                stamp=900.0 + index,
                wake=True,
            )
        _session(tmp_path, "subject00001", created=999.0, stamp=2_000.0)

        rank_head = catalogue_page(tmp_path, limit=5)
        activity_head = catalogue_page(tmp_path, limit=5, sort=CATALOGUE_SORT_ACTIVITY)

        rank_ids = [entry.id for entry in rank_head.entries]
        activity_ids = [entry.id for entry in activity_head.entries]
        assert "subject00001" not in rank_ids
        assert activity_ids[0] == "subject00001"

    def test_an_unknown_sort_is_refused_by_the_page(self, tmp_path: Path) -> None:
        """The route refuses by name before the store; a direct caller gets a
        hard failure rather than a silently different order."""
        _session(tmp_path, "plain0000001", created=1_000.0)

        with pytest.raises(ValueError, match="unknown catalogue sort"):
            catalogue_page(tmp_path, limit=5, sort="sideways")


class TestActivityCursors:
    """A position is the scope's AND the order's; a walk covers the store once."""

    def test_an_activity_walk_covers_the_store_exactly_once(self, tmp_path: Path) -> None:
        for index in range(23):
            _session(
                tmp_path,
                f"row{index:09d}",
                created=1_000.0 + index,
                stamp=1_000.0 + (index * 7) % 23,  # a shuffled activity clock
            )
        expected = [
            entry.id
            for entry in load_catalog(tmp_path, limit=200)
            if True  # every row is cold here; order still differs from birth
        ]
        big = catalogue_page(tmp_path, limit=200, sort=CATALOGUE_SORT_ACTIVITY)
        expected = [entry.id for entry in big.entries]
        assert len(expected) == 23

        seen: list[str] = []
        cursor: str | None = None
        pages = 0
        while True:
            page = catalogue_page(
                tmp_path, cursor=cursor, limit=5, sort=CATALOGUE_SORT_ACTIVITY
            )
            pages += 1
            seen += [entry.id for entry in page.entries]
            assert (page.next_cursor is not None) == (len(page.entries) == 5)
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
            assert pages < 10, "the walk did not terminate"

        assert pages == 5, "23 rows at 5 per page is five pages"
        assert seen == expected
        assert len(set(seen)) == len(seen), "a walk must not repeat a row"

    @pytest.mark.parametrize("replay_order", [RANK_HEAD, CATALOGUE_SORT_ACTIVITY])
    def test_a_cursor_from_the_other_order_answers_the_first_page(
        self, tmp_path: Path, replay_order: str
    ) -> None:
        """Both directions: the token names its order, and a mismatch is a
        first page -- the documented remedy for a token that cannot be used."""
        for index in range(6):
            _session(tmp_path, f"row{index:09d}", created=1_000.0 + index)

        minted_order = CATALOGUE_SORT_ACTIVITY if replay_order == RANK_HEAD else RANK_HEAD
        first = catalogue_page(tmp_path, limit=2, sort=minted_order)
        assert first.next_cursor is not None

        replayed = catalogue_page(tmp_path, cursor=first.next_cursor, limit=2, sort=replay_order)
        fresh = catalogue_page(tmp_path, limit=2, sort=replay_order)

        assert replayed.cursor_missing is True
        assert [entry.id for entry in replayed.entries] == [
            entry.id for entry in fresh.entries
        ]

    def test_a_v2_token_without_its_order_declaration_is_unusable(self, tmp_path: Path) -> None:
        """The declaration is part of the position's meaning, so the key is
        read only under it -- a token missing ``o`` is answered as a first
        page, never resumed under an assumed order."""
        _session(tmp_path, "plain0000001", created=1_000.0)
        forged = base64.urlsafe_b64encode(
            json.dumps(
                {"v": 2, "s": None, "k": [0, 1_000.0, "plain0000001"]},
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).decode("ascii").rstrip("=")

        assert decode_cursor(forged) is None


class TestSiblingPathsUnderTheActivityOrder:
    """The paths the fix must not disturb, asserted in the new order's terms.

    The order is a KEY, not a filter: every membership decision (origin
    hiding, archive, bindings, subagent layers) happens on the candidate list
    BEFORE either sort runs, so one order cannot see rows the other cannot.
    These two tests are the guards for the sibling paths the change is most
    likely to be blamed for -- a hidden configuration run and an ordinary
    team-attached conversation -- rather than a re-derivation of the scan's
    own suite (`test_catalog_scan_cost` owns that).
    """

    def test_an_origin_hidden_run_is_in_no_order(self, tmp_path: Path) -> None:
        """agent-config runs stay hidden from EVERY listing, deliberately.

        The origin is absent from ``USER_ORIGINS``, and that absence IS the
        hiding rule (``resume.is_user_origin``), so a fresh run of the new
        order must not resurrect one -- hiding is membership, and membership
        is chosen before either key.
        """
        _session(tmp_path, "plain0000001", created=1_000.0, stamp=1_500.0)
        hidden = _session(tmp_path, "config0000001", created=1_001.0, stamp=1_600.0)
        (hidden / "origin.json").write_text(
            json.dumps({"origin": "agent-config"}), encoding="utf-8"
        )

        for order in (RANK_HEAD, CATALOGUE_SORT_ACTIVITY):
            page = catalogue_page(tmp_path, limit=50, sort=order)
            ids = [entry.id for entry in page.entries]
            assert "config0000001" not in ids, f"hidden under sort={order!r}"
            assert "plain0000001" in ids, f"the visible row is carried under sort={order!r}"

    def test_a_team_attached_conversation_still_leads_the_activity_head(
        self, tmp_path: Path
    ) -> None:
        """A team-started plain conversation is an ordinary row to the order.

        Its binding is written through the store's real writer, because the
        claim is about what the catalogue does with what the store holds: the
        activity key reads membership and the activity clock, never the
        binding -- a scope may FILTER by it (``in_scope``), and no order may
        drop a row for having one.
        """
        from local_operator.resume import write_session_attachment

        for index in range(10):
            _session(
                tmp_path,
                f"sched{index:08d}",
                created=500.0 + index,
                stamp=500.0 + index,
                wake=True,
            )
        subject = _session(tmp_path, "teamsubject1", created=900.0, stamp=1_500.0)
        write_session_attachment(subject, team="lopdev", agent="", goal="")

        rank_head = catalogue_page(tmp_path, limit=10)
        activity_head = catalogue_page(tmp_path, limit=10, sort=CATALOGUE_SORT_ACTIVITY)

        assert "teamsubject1" not in [entry.id for entry in rank_head.entries]
        assert [entry.id for entry in activity_head.entries][0] == "teamsubject1"
