"""The sidebar and phone keep immutable targets within each outcome category."""

from __future__ import annotations

import os
from dataclasses import replace
from unittest.mock import patch

import pytest

from local_operator.mobile.daemon import SessionEntry, SessionTable
from local_operator.mobile.types import SessionProjection, SessionRecord
from local_operator.resume import SessionRow
from local_operator.tui.app import OperatorApp
from local_operator.tui.session_catalog import CatalogEntry, rank_entries
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from tests.unit.tui.test_session_sidebar import _quiesce_sidebar_refresh


def entries():
    rows = []
    for category, kind, busy, pending in [
        ("gate", "complete", True, "ask"),
        ("done", "complete", False, None),
        ("error", "error", False, None),
        ("interrupted", "interrupted", False, None),
        ("busy", "error", True, None),
        ("idle", "", False, None),
    ]:
        for suffix, created in [("old", 1), ("b", 2), ("a", 2)]:
            sid = f"{category}-{suffix}"
            rows.append(
                CatalogEntry(
                    SessionRow(
                        sid,
                        100 - created,
                        sid,
                        live_state="busy" if busy else "idle",
                        pending=pending,
                        created_at=created,
                    ),
                    unseen=bool(kind),
                    completion_kind=kind,
                )
            )
    return rows


def expected():
    return [
        f"{kind}-{suffix}"
        for kind in ("gate", "done", "error", "interrupted", "busy", "idle")
        for suffix in ("a", "b", "old")
    ]


def test_durable_catalog_loads_birth_even_when_transcript_activity_changes(tmp_path):
    import json

    from local_operator.tui.session_catalog import cached_session_rows

    for sid, created in [("old", 10), ("new", 20)]:
        directory = tmp_path / "sessions" / sid
        directory.mkdir(parents=True)
        (directory / "created_at.json").write_text(str(created))
        (directory / "transcript.jsonl").write_text(
            json.dumps(
                {
                    "id": sid,
                    "ts": 1,
                    "type": "message",
                    "payload": {"role": "user", "content": [{"type": "text", "text": sid}]},
                }
            )
            + "\n"
        )
    for tick in range(4):
        os.utime(tmp_path / "sessions" / "old" / "transcript.jsonl", (9999 + tick, 9999 + tick))
        rows = cached_session_rows(tmp_path)
        assert {row.id: row.created_at for row in rows} == {"old": 10, "new": 20}
        assert [e.id for e in rank_entries([CatalogEntry(row) for row in rows])] == ["new", "old"]


def test_category_birth_id_order_ignores_activity_and_input_order():
    rows = entries()
    assert [e.id for e in rank_entries(rows)] == expected()
    for tick in range(8):
        rows = [
            replace(e, row=e.row._replace(mtime=(index + tick) * 777))
            for index, e in enumerate(reversed(rows))
        ]
        assert [e.id for e in rank_entries(rows)] == expected()
    assert all(e.active for e in rows)
    cold = CatalogEntry(SessionRow("cold", 9999, "Viewed history", created_at=3))
    assert not cold.active
    assert rank_entries([cold, *rows])[-1] == cold


def wake_rows():
    """Cold rows whose armed wakes are deliberately the OLDEST in the group.

    Birth date is the only other key inside Previous, so making the armed rows
    the oldest is what forces the assertion to be about the wake: under the
    previous ``(tier, -birth, id)`` they sorted LAST, which is the behaviour
    reported.
    """
    return [
        CatalogEntry(SessionRow("plain-new", 100, "Newest, no wake", created_at=90)),
        CatalogEntry(
            SessionRow(
                "dormant-mid",
                100,
                "Stopped, wake dormant",
                created_at=50,
                wakes=1,
                wakes_dormant=True,
            )
        ),
        CatalogEntry(SessionRow("plain-old", 100, "Oldest, no wake", created_at=30)),
        CatalogEntry(SessionRow("armed-b", 100, "Armed wake, older", created_at=20, wakes=1)),
        CatalogEntry(SessionRow("armed-a", 100, "Armed wake, oldest", created_at=10, wakes=2)),
    ]


# ``(live_state, pending, wakes, wakes_dormant)`` -> the band ``rank`` assigns.
# Band 0/1 are reachable ONLY from a cold row; every active row takes the same
# constant 2 whatever its wake. Pinning the band rather than only an order is
# what makes a regression name itself: an order-only assertion reports "the list
# is wrong", this reports that an active row became distinguishable at all.
WAKE_BANDS = [
    # Active rows: the wake is inert, whatever the state or the timer.
    ("attached", None, 0, False, 2),
    ("attached", None, 2, False, 2),
    ("attached", None, 1, True, 2),
    ("busy", None, 0, False, 2),
    ("busy", None, 2, False, 2),
    ("wedged", None, 0, False, 2),
    ("wedged", None, 2, False, 2),
    ("idle", None, 0, False, 2),
    ("idle", None, 2, False, 2),
    ("idle", None, 1, True, 2),
    # A PENDING row is active even with an empty live_state -- the case a
    # "presence outranks a wake" guard silently misses.
    ("", "ask", 2, False, 2),
    ("", "approval", 2, False, 2),
    ("", "ask", 0, False, 2),
    # Cold history, the only group the key speaks about.
    ("", None, 2, False, 0),
    ("", None, 1, True, 1),
    ("", None, 0, False, 2),
]


@pytest.mark.parametrize("live_state,pending,wakes,dormant,band", WAKE_BANDS)
def test_wake_rank_is_scoped_to_cold_rows(live_state, pending, wakes, dormant, band):
    entry = CatalogEntry(
        SessionRow(
            "s",
            100,
            "row",
            live_state=live_state,
            pending=pending,
            wakes=wakes,
            wakes_dormant=dormant,
            created_at=1,
        )
    )
    assert entry.rank[1] == band
    # The structural claim behind the scoping: an active row ALWAYS lands on the
    # constant, so no wake can make one distinguishable from another.
    if entry.active:
        assert entry.rank[1] == 2


# ``(case, rows, expected)`` for the three inversions a uniform key produced.
# Each was reproduced independently before the key was scoped; they are the
# regression tests for tiers 5, 4 and 0 respectively.
#
# In every case the ARMED row is the OLDER one, so birth date alone already puts
# it second and the wake is the only thing that could lift it. That construction
# is what makes these able to fail: with the armed row newest they would pass
# under either key and pin nothing.
ACTIVE_TIER_CASES = [
    (
        "tier5-attached-above-armed-idle",
        [
            ("idle-armed", "idle", None, 2, False, 10),
            ("attached", "attached", None, 0, False, 90),
        ],
        ["attached", "idle-armed"],
    ),
    (
        "tier4-wedged-above-armed-busy",
        [
            ("busy-timer", "busy", None, 3, False, 20),
            ("wedged", "wedged", None, 0, False, 50),
        ],
        ["wedged", "busy-timer"],
    ),
    (
        "tier0-plain-pending-above-armed-pending",
        [
            ("pending-armed", "", "ask", 2, False, 10),
            ("pending-plain", "", "approval", 0, False, 90),
        ],
        ["pending-plain", "pending-armed"],
    ),
]


@pytest.mark.parametrize(
    "case,rows,expected", ACTIVE_TIER_CASES, ids=[c[0] for c in ACTIVE_TIER_CASES]
)
def test_an_armed_wake_never_reorders_an_active_tier(case, rows, expected):
    """The INVERSE property: inside Active, the wake must change nothing.

    A uniform wake key looked free because it "only breaks ties within one
    category". Rows inside a tier are not interchangeable, though: tier 5 mixes
    ``attached`` with ``idle``, tier 4 mixes ``busy`` with ``wedged``, and tier 0
    mixes the two gates. Each inversion below was reproduced against the uniform
    key, and the tier-0 one is the reason the fix is scoping rather than a
    presence check — a pending row can carry an empty ``live_state``.
    """
    entries = [
        CatalogEntry(
            SessionRow(
                sid,
                100,
                sid,
                live_state=state,
                pending=pending,
                wakes=wakes,
                wakes_dormant=dormant,
                created_at=created,
            )
        )
        for sid, state, pending, wakes, dormant, created in rows
    ]
    assert [e.id for e in rank_entries(entries)] == expected
    # Same order with the wakes stripped: proof the wake is inert here rather
    # than merely losing to another key that happens to agree today.
    stripped = [replace(e, row=e.row._replace(wakes=0, wakes_dormant=False)) for e in entries]
    assert [e.id for e in rank_entries(stripped)] == expected
    assert len({e.rank[1] for e in entries}) == 1


def test_armed_wake_floats_to_top_of_previous_above_dormant():
    rows = wake_rows()
    # Armed rows first (the oldest-birth ones, so this cannot be a birth-date
    # artefact), then the dormant row, then the plain rows by birth date.
    assert [e.id for e in rank_entries(rows)] == [
        "armed-b",
        "armed-a",
        "dormant-mid",
        "plain-new",
        "plain-old",
    ]


def test_every_clock_glyph_is_contiguous_with_armed_above_dormant():
    """The block boundary must land where the GLYPH changes, not inside it.

    A dormant wake still ranks below every armed one — it is not a future that is
    coming — but it ranks directly UNDER them rather than nowhere. The dim-vs-
    muted separation between two ``WAKE_MARKER`` glyphs measures 1.77:1 at this
    UI's 8x17px cell, which is below any discrimination threshold, so a dormant
    row stranded among the plain rows is an identical-looking glyph mid-list and
    reads as a broken sort rather than as a distinct state.
    """
    order = rank_entries(wake_rows())
    marked = [index for index, entry in enumerate(order) if entry.row.wakes]
    assert marked == list(range(marked[0], marked[0] + len(marked)))
    armed = [index for index in marked if not order[index].row.wakes_dormant]
    dormant = [index for index in marked if order[index].row.wakes_dormant]
    assert max(armed) < min(dormant)


def test_armed_wake_floats_without_changing_section_membership():
    """Floating is an ORDER change only; a cold row must stay under Previous."""
    rows = wake_rows()
    assert not any(e.active for e in rows)
    assert all(not e.active for e in rank_entries(rows))
    # And it must not overtake a genuinely active row: the tier still leads.
    live = CatalogEntry(SessionRow("live", 100, "Resident", created_at=1, live_state="idle"))
    assert rank_entries([*rows, live])[0].id == "live"


def test_armed_wake_order_is_stable_across_activity_churn():
    """Same invariant as the category test: a refresh must not move a target."""
    rows = wake_rows()
    # Pinned literally rather than read back from the first sort, so this also
    # fails if the wake key stops applying — a dynamically derived expectation
    # would agree with any order at all as long as it never changed.
    expected_order = ["armed-b", "armed-a", "dormant-mid", "plain-new", "plain-old"]
    assert [e.id for e in rank_entries(rows)] == expected_order
    for tick in range(8):
        rows = [
            replace(e, row=e.row._replace(mtime=(index + tick) * 777))
            for index, e in enumerate(reversed(rows))
        ]
        assert [e.id for e in rank_entries(rows)] == expected_order


def test_armed_wake_ties_fall_back_to_birth_then_id():
    """The wake key only breaks ties; it never replaces the durable ordering."""
    rows = [
        CatalogEntry(SessionRow("w-old", 100, "Armed, older", created_at=5, wakes=1)),
        CatalogEntry(SessionRow("w-b", 100, "Armed, newer b", created_at=9, wakes=3)),
        CatalogEntry(SessionRow("w-a", 100, "Armed, newer a", created_at=9, wakes=1)),
    ]
    assert [e.id for e in rank_entries(rows)] == ["w-a", "w-b", "w-old"]


def test_mobile_uses_same_categories_birth_and_ties():
    table = SessionTable()
    durable = {}
    for index, e in enumerate(entries()):
        entry = SessionEntry(
            SessionRecord(
                pid=900000 + index,
                kind="tui",
                session_id=e.id,
                conversation_name=e.id,
                cwd="/synthetic",
                model_label="demo",
                control_port=1,
                control_key="test",
            )
        )
        entry.projection = SessionProjection(
            session_id=e.id, pid=entry.record.pid, kind="tui", streaming=e.row.live_state == "busy"
        )
        if e.row.pending:
            from local_operator.mobile.types import PendingRequest

            entry.projection.pending = PendingRequest(
                kind="ask", request_id="test", title="Question"
            )
        table.entries[entry.record.pid] = entry
        table._attention_states[f"session/{e.id}"] = {"unseen": e.unseen, "kind": e.completion_kind}
        durable[e.id] = e.row
    for tick in range(8):
        durable = {
            sid: row._replace(mtime=tick * 1000 + i)
            for i, (sid, row) in enumerate(reversed(list(durable.items())))
        }
        for entry in table.entries.values():
            entry.record.heartbeat_at += 1000
            entry.record.started_at += 1000
        assert [r["session_id"] for r in table._merge_summaries(durable)] == expected()


@pytest.mark.asyncio
async def test_activity_refresh_keeps_mouse_target_and_keyboard_cursor(tmp_path, monkeypatch):
    for key in tuple(os.environ):
        if key.startswith("CMUX_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
    monkeypatch.setattr(OperatorApp, "_check_for_update", lambda self: None)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 45)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+b")
        _quiesce_sidebar_refresh(app)
        sidebar = app._session_sidebar
        rows = entries()
        sidebar.set_entries(rows)
        sidebar.cursor_id = "busy-b"
        await pilot.pause()
        target_y = next(
            y
            for y in range(sidebar.size.height)
            if (entry := sidebar._entry_at(y)) is not None and entry.id == "busy-b"
        )
        for tick in range(5):
            rows = [
                replace(e, row=e.row._replace(mtime=1000 * tick + i))
                for i, e in enumerate(reversed(rows))
            ]
            sidebar.set_entries(rows)
            await pilot.pause()
            assert sidebar.cursor_id == "busy-b"
            target = sidebar._entry_at(target_y)
            assert target is not None and target.id == "busy-b"
        selected = []
        with patch.object(app._sidebar_navigation, "select", side_effect=selected.append):
            await pilot.click("#session-sidebar", offset=(8, target_y))
            await pilot.pause()
        assert selected == ["busy-b"]
