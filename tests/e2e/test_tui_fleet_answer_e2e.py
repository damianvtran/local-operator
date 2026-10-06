"""End-to-end: the TUI answers a FLEET row belonging to a STOPPED session.

Why this file exists
--------------------

The TUI's fleet scope reads the cross-session index, so its rows address
conversations this app does not host. Answering one is therefore not the session
list's path: it goes through the engage seam — the same
``engage_session_client`` + ``AskErrand`` arm the phone's cold reply uses — and
the proof that matters is that a real runtime is ENGAGED and the response row
lands in that other conversation's transcript.

``tests/unit/tui/test_ask_fleet.py`` pins the routing,
``tests/e2e/test_ask_cold_answer_e2e.py`` proves the engage arm through the
relay's door, and this cell is the third: the operator's own door — the real
app, the real list, the real card — driving an answer at a row whose session is
stopped, with the engaged runtime a real child process.

WHAT IS REAL: the app and its stylesheet (``run_test``), the fleet scope's own
index read, the mounted list and the mounted card's own settle path, the durable
``asks.jsonl``/index/transcript on disk, the engage and the runtime child it
spawns, and the boot reconcile that delivers. The FLEET conversation is seeded by
booting a real session for one scripted ``ask`` turn and then disposing it, so
both the log and the derived index entry are the product's own writes rather than
a fixture's idea of them.

WHAT IS DOUBLED: the ADOPTED conversation (``FakeSession``). It is not the
subject — the subject is a row belonging to somebody else — and standing this
app's own conversation up on a scripted provider would add a second moving part
to a cell about the first.

Isolation: the autouse ``headless_tui_env`` fixture points
``LOCAL_OPERATOR_CONFIG_DIR`` at a per-test directory, so the app, the store and
the spawned child all read the scratch root and never the operator's store. The
runtime the engage leaves serving is reaped by exact pid in ``finally``.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from pathlib import Path
from typing import Any

import pytest

from local_operator.asks import policy
from local_operator.asks import store as ask_store
from local_operator.session.runtime import registry
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.ask_picker import AskPickerScreen
from local_operator.tui.widgets.ask_queue import SCOPE_FLEET, AskQueueList
from tests.e2e.harness import (
    ScriptedStream,
    build_session,
    dispose_quietly,
    text_turn,
    tool_call_turn,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory

# The e2e suite's marker plus this module's own: the cell drives the real app
# through Textual's pilot, so it is a coroutine.
pytestmark = [pytest.mark.e2e, pytest.mark.asyncio]

#: The OTHER conversation — seeded by the product, then STOPPED with an open ask.
FLEET_SESSION = "fleetdrive01"
QUESTION = "answer the fleet row?"

#: Generous, because it covers a real process spawn and a real boot reconcile on
#: a fleet running dozens of suites at once (AGENTS.md). The bound turns a hang
#: into a named failure rather than a silent stall.
BOUND_S = 150.0

#: The app's own loop is the pump; this only paces the observable checks between
#: pumps. Nothing here waits on the clock.
POLL_S = 0.05


@pytest.fixture(autouse=True)
def _the_queue_is_live(monkeypatch: pytest.MonkeyPatch) -> None:
    """The queued arm, stated rather than inherited (the sibling e2e file's rule)."""
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    # The ask gate is pinned OFF for the same reason as the sibling queue file:
    # the seeding run is scripted for the queue's own calls only, and the
    # gate's forked check is an extra provider request its tape does not carry.
    # The gate's real-path cells (where the fork IS scripted) are
    # tests/e2e/test_ask_gate_e2e.py.
    monkeypatch.setattr(policy, "ASK_GATE", False)


async def _never_answers(questions: list[Any]) -> dict[str, list[str]] | None:
    raise AssertionError("the queued path must not call the host hook")


async def _seed_stopped_fleet_session(config_dir: Path) -> tuple[Path, str]:
    """A STOPPED conversation with one open queued ask, produced by the product.

    Booting a real session for one scripted ``ask`` turn is what makes the row
    real: the ask LOG and the derived INDEX entry are both written by the queue
    itself, so the fleet scope's read finds what a live runtime left behind when
    it stopped — which is the state the operator's report is about. Hand-writing
    the index would have skipped the very write this cell exists to lean on.
    """
    (config_dir / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )
    directory = config_dir / "sessions" / FLEET_SESSION
    stream = ScriptedStream(
        [
            tool_call_turn(
                text="asking",
                tool_name="ask",
                tool_call_id="ask-fleet",
                arguments={
                    "questions": [
                        {
                            "id": "q1",
                            "question": QUESTION,
                            # TWO options, deliberately: a one-option choice is
                            # a validation error the tool reports instead of
                            # queueing, and a cell that never queued would look
                            # like an engage failure.
                            "options": [{"label": "yes"}, {"label": "no"}],
                        }
                    ]
                },
            ),
            text_turn("working"),
        ]
    )
    session = build_session(directory, stream, cwd=config_dir)
    session.set_ask_handler(_never_answers)
    try:
        await session.prompt("go")
    finally:
        await dispose_quietly(session)
    (ask_id,) = ask_store.ask_ids(ask_store.read_events(directory))
    return directory, ask_id


def _reap() -> None:
    """Kill every runtime this cell engaged, by exact pid.

    An engage deliberately leaves its runtime serving (that is what an engage
    is), so a cell that did not reap would leave a real child holding the
    transcript lease on the fleet. The scan is scoped to the registry this
    config root owns — never a bare program-name kill, which the team brief
    forbids outright.
    """
    for record, _state in registry.scan():
        pid = getattr(record, "pid", None)
        if isinstance(pid, int) and pid > 0:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)


async def _wait_on_pilot(pilot, predicate, *, timeout_s: float, what: str) -> None:
    """Advance the app's own loop until ``predicate`` holds, bounded and loud.

    The engage, the child's boot reconcile and the delivery all happen OUTSIDE
    this process's task graph, so what is awaited is the OBSERVABLE — a
    transcript row, a log event — with the pilot's own pause as the pump. The
    timeout is a failure bound, not a wait.
    """
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await pilot.pause()
        await asyncio.sleep(POLL_S)
    raise AssertionError(f"timed out after {timeout_s}s waiting for {what}")


def _answered_events(session_dir: Path, ask_id: str) -> bool:
    return any(
        event.get("kind") == ask_store.EVENT_ANSWERED and event.get("ask_id") == ask_id
        for event in ask_store.read_events(session_dir)
    )


def _response_row(session_dir: Path, ask_id: str) -> bool:
    path = session_dir / "transcript.jsonl"
    if not path.exists():
        return False
    return ask_store.response_row_id(ask_id) in path.read_text(encoding="utf-8")


async def test_a_fleet_row_on_a_stopped_session_is_answered_through_the_engage_seam(
    headless_tui_env: Path,
) -> None:
    """The whole door: fleet scope → the row → the card → an engaged delivery.

    The assertions are the three an engage must produce and a mock cannot fake:
    the durable log records the winner for THAT ask, the OTHER conversation's
    transcript gains its response row, and the index folds the ask out of the
    outstanding set the marks and the totals are computed from.
    """
    other, ask_id = await _seed_stopped_fleet_session(headless_tui_env)
    assert not (other / ".session.pid").exists(), "the fleet session must start STOPPED"
    # The row the list will paint is the product's own left-behind index entry.
    assert ask_id in [row["ask_id"] for row in ask_store.index_asks(headless_tui_env)]

    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    try:
        async with app.run_test(size=(130, 30)) as pilot:
            app._session = session
            await pilot.pause()

            # (1) THE DOOR. The fleet scope is opened the way the sidebar's note
            # opens it — the app's own action, its own off-thread index read —
            # and the list that mounts is the ONE list, on the fleet scope.
            app.action_open_fleet_asks()
            await _wait_on_pilot(
                pilot,
                lambda: bool(app.query(AskQueueList)),
                timeout_s=30,
                what="the fleet list to mount",
            )
            listing = app.query_one(AskQueueList)
            assert listing._scope == SCOPE_FLEET
            row = next((item for item in listing.rows if item.ask_id == ask_id), None)
            assert row is not None, [item.ask_id for item in listing.rows]

            # (2) THE ROW. It carries its OWN conversation, and the app's routing
            # agrees that this is not the conversation it hosts — the operator's
            # ask 2, which is what sends the answer down the engage seam.
            assert row.session_id == FLEET_SESSION
            assert app._fleet_answer_starts_here(row) is True

            # Selecting it is the widget's own gesture, so the Picked message
            # travels the path a click or Enter travels.
            listing.select(listing.visible_rows.index(row))
            listing.action_pick()
            await _wait_on_pilot(
                pilot,
                lambda: bool(app.query(AskPickerScreen)),
                timeout_s=15,
                what="the row's card to mount",
            )

            # Nothing has been delivered yet, so the two waits below cannot
            # pass on a row that was already there.
            assert not _answered_events(other, ask_id)
            assert not _response_row(other, ask_id)

            # (3) THE ANSWER, through the card's own settle path — what the
            # user's Enter on the last question runs — with nothing doubled
            # between it and the engage.
            app.query_one(AskPickerScreen).settle({"q1": ["yes"]})

            await _wait_on_pilot(
                pilot,
                lambda: _answered_events(other, ask_id),
                timeout_s=BOUND_S,
                what="the answered event for the fleet ask",
            )
            await _wait_on_pilot(
                pilot,
                lambda: _response_row(other, ask_id),
                timeout_s=BOUND_S,
                what="the response row in the OTHER conversation's transcript",
            )

            # The fold the marks and the totals are computed from no longer owes
            # it, so the mark on that row clears on the next poll.
            outstanding = ask_store.outstanding_asks(ask_store.index_asks(headless_tui_env))
            assert ask_id not in [item["ask_id"] for item in outstanding]
            settled = next(
                (
                    item
                    for item in ask_store.index_asks(headless_tui_env)
                    if item["ask_id"] == ask_id
                ),
                None,
            )
            if settled is not None:
                # Derived, and rewritten on the owner's next publish, so it may
                # lag one write — but when it IS there it must say answered.
                assert settled["status"] == ask_store.STATUS_ANSWERED
    finally:
        _reap()
