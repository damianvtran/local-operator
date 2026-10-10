"""The hidden internal wake kinds, and every human surface that must subtract them.

The ask queue rides the wake engine for its deadlines (design
``docs/design/ask-nonblocking.md`` §2.2, D10), which means an ``ask_timeout`` row
appears in the same ``schedules`` lists every wake surface reads. The failure this
file guards is the one the design's risk 4 names: a human surface that misses the
new predicate leaks a hidden timer as if it were a reminder someone set — so the
subtraction is ONE named predicate (``is_internal_wake_row``) and the surfaces
that must ask it are pinned here rather than trusted to have been found.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.wake_types import WakeSchedule
from local_operator.wakes import patience, store

ROOT = Path(__file__).resolve().parents[3]


def test_the_predicate_covers_patience_and_ask_timeout_and_nothing_else():
    assert store.is_internal_wake_row({"kind": "patience"}) is True
    assert store.is_internal_wake_row({"kind": "ask_timeout"}) is True
    assert store.is_internal_wake_row({"kind": "scheduled"}) is False
    assert store.is_internal_wake_row({}) is False
    assert store.is_internal_wake_row(None) is False


def test_the_predicate_reads_models_and_dicts_alike():
    """The session holds models and the index/supervisor hold dicts, and BOTH
    must be filtered by the same call — the review finding that produced
    ``is_patience_row``."""
    row = WakeSchedule(
        id="a-1",
        message="ask a-1 deadline",
        next_due_at=1_700_000_000_000,
        created_at=1_699_000_000_000,
        kind="ask_timeout",
        hidden=True,
    )
    assert store.is_internal_wake_row(row) is True
    assert store.is_internal_wake_row(row.model_dump()) is True


def test_an_ask_timeout_row_validates_as_a_wake_schedule():
    """The ``kind`` Literal accepts it, so a booted runtime can load the row —
    without this the queue silently degrades to its in-runtime timer."""
    row = WakeSchedule(
        id="ask-timeout-a-1",
        message="ask a-1 deadline",
        next_due_at=1_700_000_000_000,
        created_at=1_699_000_000_000,
        kind="ask_timeout",
        hidden=True,
    )
    assert WakeSchedule.model_validate(row.model_dump()).kind == "ask_timeout"


def test_a_pre_ask_build_would_drop_the_row_rather_than_misfire():
    """An OLD build's ``kind`` Literal is the two-value one, and ``extra`` is
    forbidden — so it drops the row on load and the queue falls back to its timer
    rather than firing a wake whose kind it does not know."""
    with pytest.raises(Exception):
        WakeSchedule.model_validate(
            {
                "id": "ask-timeout-a-1",
                "message": "ask a-1 deadline",
                "next_due_at": 1_700_000_000_000,
                "created_at": 1_699_000_000_000,
                "kind": "some_future_kind",
                "hidden": True,
            }
        )


def test_scheduled_rows_drops_both_internal_kinds():
    rows = [
        {"id": "w1", "kind": "scheduled", "message": "standup"},
        {"id": "patience-1", "kind": "patience", "message": "wait"},
        {"id": "ask-timeout-a-1", "kind": "ask_timeout", "message": "ask a-1 deadline"},
    ]
    assert [row["id"] for row in store.scheduled_rows(rows)] == ["w1"]
    assert [row["id"] for row in patience.scheduled_rows(rows)] == ["w1"]


def test_the_patience_module_re_exports_the_one_predicate():
    """``wakes/patience.py`` re-exports both, so a caller that already imports the
    patience engine gets the union rather than a second spelling."""
    assert patience.is_internal_wake_row({"kind": "ask_timeout"}) is True
    assert patience.is_internal_wake_row({"kind": "scheduled"}) is False


def test_the_narrow_predicate_still_means_patience_only():
    """The cap, the cancel path and the watermark delivery are ABOUT patience; a
    union there would make the ask queue's timer cancellable by ``patience``."""
    assert store.is_patience_row({"kind": "patience"}) is True
    assert store.is_patience_row({"kind": "ask_timeout"}) is False
    assert patience.is_patience_row({"kind": "ask_timeout"}) is False


#: EVERY reader of a wake scheduler's ``.schedules`` list, as
#: ``module -> {function: why it may read the RAW list}``.
#:
#: The pin is here because the failure mode is a MISSED call site — and because a
#: source test that only asserted the string ``"scheduled_rows"`` appeared in six
#: files would pass while someone added a SEVENTH listing surface beside them,
#: filtering by hand (review round 1, MINOR 8). A new reader therefore fails this
#: test until it is classified, exactly as a new path-shaped constant fails the
#: copy-set guard until it is: either it is a human surface and uses
#: ``scheduled_rows``/``is_internal_wake_row``, or it reads the raw list for one
#: of the reasons below and says so.
_SCHEDULES_READERS: dict[str, dict[str, str]] = {
    "local_operator/session/runtime/serving.py": {
        "_wake_slash": "a MUTATION path: it rebuilds the whole list to edit one row",
        "next_wake_due_at": "the supervisor's due-time read: hidden rows must still fire",
    },
    "local_operator/session/session.py": {
        "_aida_reconcile_now": "full-list writer (Aida's cadence reconcile)",
        "_apply": "full-list writer (the ask deadline row's arm/retire)",
        "_catchup_notify_from_live_rows": (
            "an engine reader: the catch-up delivery's notify intent, re-read "
            "from the live rows at take time; it renders nothing"
        ),
        "_flush_patience_armed_after": "full-list writer (patience's armed-after patch)",
        "_rebuild_wake_index_entry": "full-list writer (the index the supervisor reads)",
        "arm_ask_wake": "full-list writer (the ask deadline row)",
        "cancel_pending_patience": "full-list writer (patience's cancel)",
        "hand_wakes_to_successor": "full-list writer (the placement handoff)",
        "retire_ask_wake": "full-list writer (the ask deadline row)",
    },
    "local_operator/tools/builtin.py": {
        "_wake_cancel": "the wake tool's mutation",
        "_wake_create": "the wake tool's mutation",
        "_wake_edit": "the wake tool's mutation",
        "_wake_list": "a HUMAN surface — subtracts via ``scheduled_rows``",
        "execute_patience": "the patience engine's own door",
    },
    "local_operator/tui/widgets/wake_panel.py": {
        "sync": "a HUMAN surface — subtracts via ``scheduled_rows``",
    },
    "local_operator/wakes/patience.py": {
        "arm_patience": "the patience engine",
        "cancel_patience": "the patience engine",
    },
}

#: The files that read the list only to render it for a person, and must
#: therefore never do so unfiltered. Separate from the map above because the two
#: answer different questions: that one says WHERE the list is read, this one says
#: what a reader that a human sees must do about it.
_HUMAN_SURFACES = (
    "local_operator/tools/builtin.py",
    "local_operator/tui/widgets/wake_panel.py",
    "local_operator/server/utils/desktop_feed.py",
    "local_operator/server/routes/desktop_wakes.py",
    "local_operator/cli.py",
    "local_operator/session/catalog.py",
)


def _schedules_readers() -> dict[str, dict[str, str]]:
    """``module -> {function: base spelling}`` for every ``X.schedules`` read.

    Filtered to the base names that denote a WAKE scheduler (the word ``wake``,
    ``scheduler`` or ``sched`` in the spelling), because ``.schedules`` is also
    the agent scheduler's and a frontend view object's attribute and those are not
    this subsystem's list at all.
    """
    found: dict[str, dict[str, str]] = {}
    for path in sorted((ROOT / "local_operator").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover — a broken tree is flake8's failure
            continue
        module = str(path.relative_to(ROOT))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for inner in ast.walk(node):
                if not isinstance(inner, ast.Attribute) or inner.attr != "schedules":
                    continue
                base = ast.unparse(inner.value)
                if not any(token in base.lower() for token in ("wake", "scheduler", "sched")):
                    continue
                found.setdefault(module, {})[node.name] = base
    return found


def test_every_scheduler_reader_is_classified():
    """A NEW reader of the wake list fails here until it is classified.

    Both directions, like the copy-set guard: an unlisted reader is a surface
    nobody has decided about, and a listed reader that no longer exists is a stale
    row rather than a statement. The failure text says which of the two it is and
    which side a listing surface belongs on.
    """
    found = _schedules_readers()
    unclassified = sorted(
        f"{module}::{name}"
        for module, names in found.items()
        for name in names
        if name not in _SCHEDULES_READERS.get(module, {})
    )
    assert unclassified == [], (
        "these functions read a wake scheduler's ``.schedules`` list and no row of "
        "_SCHEDULES_READERS accounts for them:\n  "
        + "\n  ".join(unclassified)
        + "\n\nIf the reader renders the list for a PERSON, it must subtract the hidden "
        "internal timers (``wakes.store.scheduled_rows`` / ``is_internal_wake_row``) "
        "rather than filtering by hand. If it is a mutation, an engine reader or a "
        "delivery path, add it to _SCHEDULES_READERS with the reason."
    )
    stale = sorted(
        f"{module}::{name}"
        for module, names in _SCHEDULES_READERS.items()
        for name in names
        if name not in found.get(module, {})
    )
    assert stale == [], (
        "_SCHEDULES_READERS names readers that no longer read ``.schedules`` "
        "(the row is stale, not a statement):\n  " + "\n  ".join(stale)
    )


def test_every_listing_surface_goes_through_the_one_filter():
    """The human surfaces, asserted directly: each names the ONE filter.

    The sweep above cannot tell a filtered read from an unfiltered one — it sees
    the attribute either way — so the two tests are complements: this one pins what
    the human surfaces must do, and the sweep pins that no fourth surface appears
    without someone deciding which it is.
    """
    for relative in _HUMAN_SURFACES:
        assert "scheduled_rows" in (ROOT / relative).read_text(encoding="utf-8"), relative


def test_the_session_subtracts_the_union_from_the_visible_catch_up_fold():
    """``Session._prepare_missed_wake_catchup`` is the one human-visible fold that
    filtered patience directly; an ``ask_timeout`` fire reaching it would render a
    hidden timer as a wake the user set."""
    from local_operator.session import session as session_module

    source = inspect.getsource(session_module.Session._prepare_missed_wake_catchup)
    assert "is_internal_wake_row" in source
    assert "is_patience_row" not in source


def test_hidden_rows_do_not_consume_the_user_wake_cap():
    """MINOR 7: the 16-row cap is the USER's budget, so an internal timer must not
    spend it. Eight open asks would otherwise leave a person eight wakes while the
    refusal still quoted sixteen — flag-on only, and invisible until it bit."""
    from local_operator.harness.wake import build_wake_schedule
    from local_operator.harness.wake_types import MAX_WAKE_SCHEDULES

    def internal(index: int) -> WakeSchedule:
        return WakeSchedule(
            id=f"ask-timeout-a-{index}",
            message=f"ask {index} deadline",
            next_due_at=1_700_000_000_000,
            created_at=1_700_000_000_000,
            kind="ask_timeout",
        )

    rows = [internal(index) for index in range(MAX_WAKE_SCHEDULES)]
    outcome = build_wake_schedule({"message": "standup", "in": "1h"}, rows, 1_700_000_000_000)
    assert "error" not in outcome, outcome


def test_the_cap_still_refuses_the_seventeenth_real_wake():
    """The other direction, so the fix cannot be read as "the cap is gone"."""
    from local_operator.harness.wake import build_wake_schedule
    from local_operator.harness.wake_types import MAX_WAKE_SCHEDULES

    rows = [
        WakeSchedule(
            id=f"w{index}",
            message="standup",
            next_due_at=1_700_000_000_000,
            created_at=1_700_000_000_000,
        )
        for index in range(MAX_WAKE_SCHEDULES)
    ]
    outcome = build_wake_schedule({"message": "standup", "in": "1h"}, rows, 1_700_000_000_000)
    assert "error" in outcome
    assert str(MAX_WAKE_SCHEDULES) in outcome["error"]


def test_rapid_ask_wake_arms_all_survive_the_full_list_writer():
    """M3 (review round 2): the regression pin for the arm/retire serialization.

    ``arm_ask_wake``/``retire_ask_wake`` are read-modify-write over a FULL-LIST
    writer, so a snapshot taken before an ``await`` loses every row but the last:
    three arms in a row all read the pre-yield list and the final write carried
    only ``ask-timeout-c``. The loss degrades SILENTLY to the in-runtime timer,
    which is exactly the cold durability the deadline row exists to provide.

    The real methods are called unbound on a stub: they touch only
    ``_ask_wake_lock``, ``_wake.schedules``, ``_wake.update`` and
    ``_spawn_background``, and pinning the SHIPPED code is the point — a
    re-implementation here would pass while the session lost rows. The ``update``
    double yields before it reads, which is what makes the race reachable.
    """
    from local_operator.session.session import Session

    class _Wake:
        def __init__(self) -> None:
            self.schedules: list[Any] = []
            self.writes: list[list[str]] = []

        async def update(self, rows: list[Any]) -> None:
            # The yield IS the defect's window: a caller that snapshots before
            # awaiting has already built its list by the time this runs.
            await asyncio.sleep(0)
            self.schedules = list(rows)
            self.writes.append([str(getattr(r, "id", "")) for r in rows])

    class _Stub:
        def __init__(self) -> None:
            self._wake = _Wake()
            self._ask_wake_lock = asyncio.Lock()
            self.tasks: list[asyncio.Task[Any]] = []

        def _spawn_background(self, coro: Any) -> Any:
            task = asyncio.ensure_future(coro)
            self.tasks.append(task)
            return task

    def row(row_id: str) -> WakeSchedule:
        return WakeSchedule(
            id=row_id,
            message=f"ask {row_id} deadline",
            next_due_at=1_700_000_000_000,
            created_at=1_700_000_000_000,
            kind="ask_timeout",
        )

    async def scenario() -> _Stub:
        stub = _Stub()
        # ``__get__`` binds the SHIPPED method to the double (the spelling the TUI
        # suites use for the same job), so the code under test is the session's
        # own rather than a copy of it.
        arm = Session.arm_ask_wake.__get__(stub)
        stub._wake.schedules = [row("user-wake")]
        # NO await between the three: the reviewer's P2 shape, and the shape the
        # in-process queue actually produces (three enqueues in one turn).
        arm(row("ask-timeout-a"))
        arm(row("ask-timeout-b"))
        arm(row("ask-timeout-c"))
        await asyncio.gather(*stub.tasks)
        return stub

    stub = asyncio.run(scenario())
    assert [str(r.id) for r in stub._wake.schedules] == [
        "user-wake",
        "ask-timeout-a",
        "ask-timeout-b",
        "ask-timeout-c",
    ]
    assert stub._wake.writes[-1] == [
        "user-wake",
        "ask-timeout-a",
        "ask-timeout-b",
        "ask-timeout-c",
    ]

    async def interleaved() -> _Stub:
        stub = _Stub()
        stub._wake.schedules = [row("user-wake"), row("ask-timeout-a")]
        Session.arm_ask_wake.__get__(stub)(row("ask-timeout-b"))
        Session.retire_ask_wake.__get__(stub)("ask-timeout-a")
        await asyncio.gather(*stub.tasks)
        return stub

    # The retire rides the same lock, so it sees the arm's write rather than the
    # list the arm started from: the LAST state wins in both orders.
    stub = asyncio.run(interleaved())
    assert [str(r.id) for r in stub._wake.schedules] == ["user-wake", "ask-timeout-b"]


def test_the_delivery_path_still_routes_ask_timeout_rows():
    """The counterpart rule: the delivery path must NOT subtract them — a hidden
    timer still has to be delivered (as a reconcile, never as a note)."""
    from local_operator.session import session as session_module

    source = inspect.getsource(session_module.Session._deliver_wake)
    assert "is_ask_timeout_row" in source
