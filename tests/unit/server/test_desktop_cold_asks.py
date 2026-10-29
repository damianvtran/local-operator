"""A cold session snapshot serves its ask rows from the DURABLE INDEX.

WHY THIS FILE EXISTS (SA-D8, design ``docs/design/ask-nonblocking.md`` §17;
local-operator-ui #1363 QA Q-1). The desktop snapshot used to serve whatever ask
fold its facade happened to hold. A facade whose owner has stopped keeps the last
canonical state on screen (``AttachedSession._go_cold``), so after a dismissal
taken with no runtime — the cold door, which writes the LOG and the derived index
and nothing else — the snapshot still said ``status: open`` / ``asks_open: 1``
for as long as it took the pool to rebuild the facade, while ``asks/<sid>.json``
and the log both said ``dismissed``. A fresh cold facade had the opposite hole:
no asks at all, so a settled receipt never reached the card.

The rule pinned here: when this bridge's facade cannot serve a verified live
owner, the ask rows come from ``<config_dir>/asks/<sid>.json`` — the reader
``GET /v1/desktop/asks`` already uses — and a LIVE owner's own frame is left
exactly as it is.

The fixture is the real ``AttachedSession`` (cold) rather than a double: the three
gates the override reads (``is_cold``, ``remote_row``, the frontend store) are its
own properties, and a double would pin a second reading of them.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.asks import store
from local_operator.server.utils import desktop_sessions as module
from local_operator.session.attached import AttachedSession
from local_operator.session.frontend_state import FrontendSessionState, PendingAskState

#: THE REAL CLOCK, deliberately: ``_cold_ask_rows`` reads the index the way the
#: route does — ``now`` omitted, so the READER's clock decides the 7 d horizon —
#: and a frozen base in 2023 would put every row past it.
BASE = int(time.time() * 1000)
HOUR_MS = 3600 * 1000
TIMEOUT_S = 3600
EXPIRES_AT = BASE + TIMEOUT_S * 1000


def _bridge_with_retained_asks(
    tmp_path: Path, *, warm: bool = False, asks_truncated: bool = False
) -> Any:
    """The bridge, over a real facade whose retained state says one ask is OPEN.

    That retained fold is the stale shape Q-1 measured: it was true when the
    runtime was alive, and nothing rewrites it after the owner goes away.
    """
    bridge = module.DesktopSessionBridge(tmp_path, "s1", str(tmp_path))
    remote = AttachedSession(
        config_dir=tmp_path, session_id="s1", takeover_factory=module._no_takeover
    )
    remote._install_frontend(
        FrontendSessionState(
            session_id="s1",
            epoch="cold-s1",
            cwd=str(tmp_path),
            asks=[
                PendingAskState(
                    ask_id="a-1",
                    created_at=BASE,
                    expires_at=EXPIRES_AT,
                    timeout_s=TIMEOUT_S,
                    status=store.STATUS_OPEN,
                    questions=[],
                )
            ],
            asks_open=1,
            asks_truncated=asks_truncated,
        )
    )
    if warm:
        # A bound viewer: ``is_cold`` is three local reads and this is the first
        # of them. Nothing else on the snapshot path touches the client.
        remote._client = cast(Any, SimpleNamespace(connected=True))
        remote._ready_for_events = True
    bridge.remote = remote
    (tmp_path / store.SESSIONS_DIRNAME / "s1").mkdir(parents=True, exist_ok=True)
    return bridge


def _dismissed_entry(root: Path, *, dismissed_at: int = BASE + 1000) -> None:
    """The index entry the COLD DOOR leaves: dismissed, stamped, with its reason."""
    store.write_entry(
        root,
        "s1",
        cwd=str(root),
        asks=[
            {
                "ask_id": "a-1",
                "created_at": BASE,
                "expires_at": EXPIRES_AT,
                "timeout_s": TIMEOUT_S,
                "urgent": False,
                "status": store.STATUS_DISMISSED,
                "delivered": True,
                "questions": [],
                "dismissed_at": dismissed_at,
                "dismissed_reason": "answered_in_chat",
            }
        ],
    )


def _snapshot_asks(snapshot: dict[str, Any]) -> tuple[Any, Any]:
    front = snapshot["payload"]["frontend"]["snapshot"]
    return front.get("asks"), front.get("asks_open")


@pytest.mark.asyncio
async def test_a_cold_snapshot_serves_the_index_instead_of_the_retained_rows(tmp_path: Path):
    """Q-1's repro: the retained fold says open, the index says dismissed."""
    bridge = _bridge_with_retained_asks(tmp_path)
    assert bridge.remote.is_cold, "the fixture must be the cold shape under test"
    _dismissed_entry(tmp_path)

    rows, open_count = _snapshot_asks(await bridge.snapshot())

    assert [row["status"] for row in rows] == [store.STATUS_DISMISSED]
    assert rows[0]["dismissed_reason"] == "answered_in_chat"
    # And the tally follows the rows: a settled row is not an ask anybody owes.
    assert open_count == 0


@pytest.mark.asyncio
async def test_a_cold_snapshot_still_says_nothing_when_the_index_has_no_entry(tmp_path: Path):
    """Absence of the entry is not a claim that the queue is empty."""
    bridge = _bridge_with_retained_asks(tmp_path)
    rows, open_count = _snapshot_asks(await bridge.snapshot())
    # The facade's own retained value is left untouched rather than replaced by
    # an invented "(no rows, 0)" — that pair is the wire's LIVE-BUT-EMPTY signal,
    # and this conversation's ask engine is not what is being described.
    assert [row["status"] for row in rows] == [store.STATUS_OPEN]
    assert open_count == 1


@pytest.mark.asyncio
async def test_a_facade_that_warms_up_during_the_read_is_not_overridden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Review round 1, m1: the verdict is re-asked before the rows are applied.

    The index read yields the loop twice, so a facade can finish a resync in that
    window. Judging "cold" only before the await would then lay index rows over
    the live fold, and a delta at or below this frame's watermark would be
    dropped as superseded while the frame carried the older rows. Here the flip
    happens INSIDE the read, which is exactly the window.
    """
    bridge = _bridge_with_retained_asks(tmp_path)
    remote = bridge.remote
    assert remote.is_cold, "the fixture must start cold"
    _dismissed_entry(tmp_path)

    real = store.session_asks

    def warming(*args: Any, **kwargs: Any) -> Any:
        rows = real(*args, **kwargs)
        # The owner answers between the gate and the apply.
        remote._client = cast(Any, SimpleNamespace(connected=True))
        remote._ready_for_events = True
        return rows

    monkeypatch.setattr(store, "session_asks", warming)

    rows, open_count = _snapshot_asks(await bridge.snapshot())

    assert not remote.is_cold
    assert [row["status"] for row in rows] == [
        store.STATUS_OPEN
    ], "the live fold must win: index rows over it are the m1 defect"
    assert open_count == 1


@pytest.mark.asyncio
async def test_a_cold_snapshot_bounds_the_index_rows(tmp_path: Path):
    """Review round 1, m2: the override runs AFTER ``sync_wire_payload`` bounded
    the facade's own list, so it has to bind its own rows too — and the tally is
    the WHOLE list's outstanding count, taken before the bound, exactly as the
    live path does."""
    bridge = _bridge_with_retained_asks(tmp_path)
    fat = "Rotate the deploy key before the cutover? " * 60
    store.write_entry(
        tmp_path,
        "s1",
        cwd=str(tmp_path),
        asks=[
            {
                "ask_id": f"a-{index}",
                "created_at": BASE,
                "expires_at": EXPIRES_AT,
                "timeout_s": TIMEOUT_S,
                "urgent": False,
                "status": store.STATUS_OPEN,
                "delivered": False,
                "questions": [{"id": "q1", "question": fat, "options": []}],
            }
            for index in range(20)
        ],
    )

    snapshot = await bridge.snapshot()
    rows, open_count = _snapshot_asks(snapshot)
    front = snapshot["payload"]["frontend"]["snapshot"]

    assert 0 < len(rows) < 20, "the wire's budget must trim the index rows"
    assert front.get("asks_truncated") is True
    assert open_count == 20, "the tally is the whole list's, not the prefix's"


@pytest.mark.asyncio
async def test_a_stale_truncation_flag_does_not_survive_the_override(tmp_path: Path):
    """The other half of m2: the flag must describe the rows THIS frame ships.

    A facade that went cold carries the last state it was handed, ``asks_truncated``
    included. Clearing it before the bound is what stops a two-row frame from
    claiming the list it holds was clipped.
    """
    bridge = _bridge_with_retained_asks(tmp_path, asks_truncated=True)
    _dismissed_entry(tmp_path)

    snapshot = await bridge.snapshot()
    rows, _ = _snapshot_asks(snapshot)
    front = snapshot["payload"]["frontend"]["snapshot"]

    assert [row["status"] for row in rows] == [store.STATUS_DISMISSED]
    assert "asks_truncated" not in front


@pytest.mark.asyncio
async def test_a_warm_snapshot_keeps_the_owners_own_rows(tmp_path: Path):
    """A live owner is authoritative: its frame carries drafts the index omits."""
    bridge = _bridge_with_retained_asks(tmp_path, warm=True)
    assert not bridge.remote.is_cold, "the fixture must be the warm shape"
    _dismissed_entry(tmp_path)

    rows, open_count = _snapshot_asks(await bridge.snapshot())

    assert [row["status"] for row in rows] == [store.STATUS_OPEN]
    assert open_count == 1
