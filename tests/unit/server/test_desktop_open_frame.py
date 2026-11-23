"""The open frame over the REAL routes: the flag, the unit, and the first frame.

The page builder's own rules are pinned in ``tests/unit/session/test_open_frame``
and the run derivation's in ``test_transcript_index``; this file is the half a
client actually meets — the query flag on all three read routes, the shape an
older renderer must still get byte for byte, and the one fact the event stream
states before the snapshot arrives.

Every case runs against a synthetic HOME, so the operator's own store is never
touched.
"""

from __future__ import annotations

import asyncio
import gc
import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.harness.types import Message, TextContent
from local_operator.server.models.desktop_sessions import HistoryPage
from local_operator.server.routes import desktop_sessions
from local_operator.server.utils.desktop_sessions import (
    DesktopSessionBridge,
    DesktopSessions,
)
from local_operator.session import index_read_model
from local_operator.session import transcript_index as ti
from local_operator.session.transcript import (
    TRANSCRIPT_FILENAME,
    Transcript,
    TranscriptEntry,
)

TOKEN = "synthetic-desktop-token"

#: How long a gated scan waits for its gate before finishing anyway. NOT a
#: latency figure: a correct frame answers while the gate is still shut, so this
#: only bounds how long a mutant that awaits the scan takes to fail.
_SCAN_HANG_S = 10.0


def _finished_without_suspending(coro: Any) -> Any:
    """Run ``coro`` one step and return its value; fail if it awaited anything.

    The open-frame read's "waits for none of it" arms are properties of control
    flow -- there is no await on those paths -- so they are checked as such,
    not as a wall-clock ceiling a loaded runner can cross. A coroutine that
    reaches any real wait yields on its first step instead of finishing.
    """
    try:
        coro.send(None)
    except StopIteration as done:
        return done.value
    coro.close()
    raise AssertionError("the read suspended: it waited on something it must not")


@pytest.fixture(autouse=True)
def _desktop_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))


def _entry(id_: str, ts: float, type_: str, payload: dict[str, Any]) -> TranscriptEntry:
    return TranscriptEntry(id_, ts, type_, payload)


def _conversation(
    turns: int, *, store: str = "spend", per_turn_tools: int = 2
) -> list[TranscriptEntry]:
    """A journal with runs, tools and the bookkeeping rows an open carries.

    ``store`` is how the spend receipts are written: as ``type: "custom"`` rows
    (the envelope this build writes) with a checkpoint row every turn, which is
    what makes the page's unit visible — a journal-entry page spends its slots on
    them, a paintable-row page does not.
    """
    rows: list[TranscriptEntry] = []
    ts = 100.0
    for turn in range(turns):
        ts += 1
        rows.append(
            _entry(
                f"start{turn:02d}",
                ts,
                "custom",
                {"custom_type": "attention_started", "details": {"token": f"t{turn}"}},
            )
        )
        ts += 1
        rows.append(
            _entry(
                f"user{turn:02d}",
                ts,
                "message",
                {"kind": "message", "role": "user", "content": [{"text": f"turn {turn}"}]},
            )
        )
        for call in range(per_turn_tools):
            ts += 1
            rows.append(
                _entry(
                    f"tool{turn:02d}{call}",
                    ts,
                    "message",
                    {
                        "kind": "message",
                        "role": "tool",
                        "content": [{"text": "output"}],
                        "provider_payload": {"details": {}, "duration_s": 2.5},
                    },
                )
            )
        ts += 1
        rows.append(
            _entry(
                f"ans{turn:02d}",
                ts,
                "message",
                {
                    "kind": "message",
                    "role": "assistant",
                    "content": [{"text": f"answer {turn}"}],
                },
            )
        )
        if store == "spend":
            ts += 1
            rows.append(
                _entry(
                    f"spend{turn:02d}",
                    ts,
                    "custom",
                    {"custom_type": "session_spend.v1", "details": {"micro": 1000}},
                )
            )
            ts += 1
            rows.append(
                _entry(
                    f"ckpt{turn:02d}",
                    ts,
                    "custom",
                    {
                        "custom_type": "frontend_state_checkpoint_v1",
                        "details": {"state": {"session_id": "seed", "sequence": turn}},
                    },
                )
            )
    return rows


def _write(directory: Path, rows: list[TranscriptEntry]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    lines = [row.to_json() for row in rows]
    (directory / TRANSCRIPT_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


class _Harness:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.app = FastAPI()
        self.app.include_router(desktop_sessions.router)
        self.pool = DesktopSessions(root)
        self.app.state.desktop_sessions = self.pool
        self.app.state.config_manager = ConfigManager(config_dir=root)
        self.session_id = ""
        self.client: AsyncClient | None = None

    async def __aenter__(self) -> "_Harness":
        self.session_id = await self.pool.create(str(self.root))
        self.client = AsyncClient(
            transport=ASGITransport(app=self.app),
            base_url="http://localhost",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self.client is not None:
            await self.client.aclose()
        ti._reset_for_tests()

    def seed(self, turns: int = 6, **kwargs: Any) -> None:
        _write(self.root / "sessions" / self.session_id, _conversation(turns, **kwargs))

    async def warm(self) -> None:
        """Build the index the way the rail's own first poll does.

        Through ``start_refresh`` rather than ``refresh_index`` on purpose: the
        frame's fast path reads the RESIDENT index, which only the refresh task
        publishes, so a test that built one with the synchronous call would be
        asserting about a cache the product never fills.
        """
        task, _started = ti.start_refresh(self.root, self.session_id)
        assert await task is not None


@pytest.mark.asyncio
async def test_without_the_flag_no_open_frame_key_is_served(tmp_path: Path) -> None:
    """THE BYTE-IDENTITY GUARD, at the level a client can be broken by.

    An older renderer must receive today's page, and today's page has exactly four
    keys. A defaulted field on the response model would have added ``"runs":
    null`` to EVERY page and quietly changed every existing client's bytes, which
    is why the model omits the three keys while they are unset.
    """
    async with _Harness(tmp_path) as harness:
        harness.seed(turns=3)
        assert harness.client is not None
        base = f"/v1/desktop/sessions/{harness.session_id}"
        page = (await harness.client.get(f"{base}/history")).json()["result"]
        assert set(page) == {"entries", "has_more", "cursor_missing", "has_newer"}
        snapshot = (await harness.client.get(base)).json()["result"]
        assert set(snapshot["payload"]["history"]) == set(page)
        assert "attaching" not in snapshot["payload"]["history"]


@pytest.mark.asyncio
async def test_the_flag_serves_the_open_frame_shape(tmp_path: Path) -> None:
    """With the flag the three keys are present, and always together."""
    async with _Harness(tmp_path) as harness:
        harness.seed(turns=3)
        assert harness.client is not None
        page = (
            await harness.client.get(
                f"/v1/desktop/sessions/{harness.session_id}/history", params={"open_frame": 1}
            )
        ).json()["result"]
        assert {"runs", "runs_state", "head_cut"} <= set(page)
        assert page["runs_state"] in {"ready", "building", "unavailable", "unsupported"}


@pytest.mark.asyncio
async def test_limit_counts_paintable_rows_not_journal_entries(tmp_path: Path) -> None:
    """THE UNIT CHANGE, which is the whole point of the flag.

    The same journal read both ways: the plain page spends its slots on the
    checkpoint and spend rows (they are journal entries), the open frame serves
    paintable rows only — so no served entry is a row the desktop cannot paint.
    """
    async with _Harness(tmp_path) as harness:
        harness.seed(turns=8)
        assert harness.client is not None
        base = f"/v1/desktop/sessions/{harness.session_id}/history"

        plain = (await harness.client.get(base, params={"limit": 10})).json()["result"]
        plain_types = [row["payload"].get("custom_type") for row in plain["entries"]]
        assert "frontend_state_checkpoint_v1" in plain_types

        frame = (await harness.client.get(base, params={"limit": 10, "open_frame": 1})).json()[
            "result"
        ]
        served_types = [row["payload"].get("custom_type") for row in frame["entries"]]
        assert "frontend_state_checkpoint_v1" not in served_types
        assert "session_spend.v1" not in served_types
        assert len(frame["entries"]) >= 10
        assert frame["head_cut"] is False or frame["entries"][0]["payload"].get("role") != "user"


@pytest.mark.asyncio
async def test_the_cut_reaches_a_run_head(tmp_path: Path) -> None:
    """The extension: the page is cut back to the run's opening user row.

    The shape is the ordinary one — a turn of one tool call and an answer — and
    the request asks for two rows, so the first page IS the run's middle. One more
    page reaches the head, and the page then holds the whole run: that is what
    lets a client condense the oldest run on it exactly, with no align walk.
    """
    async with _Harness(tmp_path) as harness:
        _write(
            harness.root / "sessions" / harness.session_id,
            [
                _entry(
                    "s1",
                    1.0,
                    "custom",
                    {"custom_type": "attention_started", "details": {"token": "t1"}},
                ),
                _entry(
                    "u1",
                    1.1,
                    "message",
                    {"kind": "message", "role": "user", "content": [{"text": "one"}]},
                ),
                _entry(
                    "x1",
                    1.2,
                    "message",
                    {"kind": "message", "role": "tool", "content": [{"text": "out"}]},
                ),
                _entry(
                    "a1",
                    1.3,
                    "message",
                    {"kind": "message", "role": "assistant", "content": [{"text": "done"}]},
                ),
                _entry(
                    "s2",
                    2.0,
                    "custom",
                    {"custom_type": "attention_started", "details": {"token": "t2"}},
                ),
                _entry(
                    "u2",
                    2.1,
                    "message",
                    {"kind": "message", "role": "user", "content": [{"text": "two"}]},
                ),
                _entry(
                    "x2",
                    2.2,
                    "message",
                    {"kind": "message", "role": "tool", "content": [{"text": "out"}]},
                ),
                _entry(
                    "a2",
                    2.3,
                    "message",
                    {"kind": "message", "role": "assistant", "content": [{"text": "done"}]},
                ),
            ],
        )
        assert harness.client is not None
        page = (
            await harness.client.get(
                f"/v1/desktop/sessions/{harness.session_id}/history",
                params={"limit": 2, "open_frame": 1},
            )
        ).json()["result"]
        assert page["head_cut"] is False
        assert [row["id"] for row in page["entries"]] == ["u2", "x2", "a2"]
        assert page["entries"][0]["payload"]["role"] == "user"


@pytest.mark.asyncio
async def test_a_head_beyond_the_budget_is_stated_rather_than_faked(tmp_path: Path) -> None:
    """A run longer than the extension can reach reports ``head_cut``.

    THIS IS THE OPERATOR'S OWN SHAPE: a settled run of hundreds of calls ends
    above the window, so no budget small enough to serve reaches its head. The
    page must not pretend otherwise — ``head_cut: true`` is what tells a renderer
    that its oldest run is a fragment — and the run's true size is exactly what
    ``runs`` carries for it.
    """
    async with _Harness(tmp_path) as harness:
        # 300 calls in the last turn: its head is further above the window than
        # the whole head-hunt budget, so this is the shape the trim exists for.
        harness.seed(turns=3, per_turn_tools=300)
        assert harness.client is not None
        page = (
            await harness.client.get(
                f"/v1/desktop/sessions/{harness.session_id}/history",
                params={"limit": 100, "open_frame": 1},
            )
        ).json()["result"]
        assert page["head_cut"] is True
        assert page["entries"][0]["payload"]["role"] != "user"
        assert page["has_more"] is True


@pytest.mark.asyncio
async def test_head_cut_is_true_when_a_cap_stops_the_extension(tmp_path: Path) -> None:
    """A run longer than the whole extension budget: the page SAYS so.

    The page is then the plain newest-N paintable rows and the run's true size
    lives in ``runs`` — which is what the client needs to draw a complete bar
    without the align walk.
    """
    async with _Harness(tmp_path) as harness:
        # One enormous run: 500 tool rows under a single user row.
        rows = _conversation(turns=1, per_turn_tools=500)
        _write(harness.root / "sessions" / harness.session_id, rows)
        assert harness.client is not None
        page = (
            await harness.client.get(
                f"/v1/desktop/sessions/{harness.session_id}/history",
                params={"limit": 100, "open_frame": 1},
            )
        ).json()["result"]
        assert page["head_cut"] is True
        assert len(page["entries"]) <= 400
        assert page["has_more"] is True


@pytest.mark.asyncio
async def test_the_first_frame_spends_only_what_is_left_of_its_budget(
    tmp_path: Path,
) -> None:
    """The facts come out of the read's own budget, never on top of it.

    THE RENDERER STILL HAS TO PAINT, so the wait is a DEADLINE
    (``OPEN_FRAME_SNAPSHOT_BUDGET_S``, measured from the request reaching the
    bridge) and not a fixed sleep: a scan that lands inside what is left answers
    READY with facts, and one that does not answers ``building`` so the client
    keeps today's path until the next frame. Both arms are stated here, and the
    second one also pins the deadline itself — a frame whose budget is already
    spent must not wait at all.

    THE READY ARM'S BUDGET IS A HANG CEILING, NOT THE PRODUCT FIGURE. "Lands
    inside what is left" is a property of the wait, not of how fast this host
    scans: with the real 120 ms the arm asserted that a cold scan of the fixture
    finishes in 120 ms on whatever runner it lands on, and a loaded CI runner's
    scan did not (CI run 43371490028, ``'building' == 'ready'``; reproduced 5/5
    with 150 ms injected into the first ``refresh_index``). A ceiling no scan of
    four turns approaches makes "the scan landed before the deadline" true by
    construction, so the arm now pins only what it means: a cold open AWAITS
    its refresh out of the deadline and answers READY with facts (drop the wait
    and it answers ``building``). The other two arms keep the real figure: they
    pin that a spent or too-small budget is not waited on.
    """
    from local_operator.server.utils import desktop_sessions as desktop_sessions_utils

    async with _Harness(tmp_path) as harness:
        harness.seed(turns=4)
        assert harness.client is not None
        url = f"/v1/desktop/sessions/{harness.session_id}/history"

        assert (
            ti.fresh_resident(harness.root, harness.session_id) is None
        ), "the ready arm must exercise the cold refresh wait, not a resident index"
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(desktop_sessions_utils, "OPEN_FRAME_SNAPSHOT_BUDGET_S", 600.0)
            first = (await harness.client.get(url, params={"limit": 5, "open_frame": 1})).json()[
                "result"
            ]
        assert first["runs_state"] == "ready"
        settled = [run for run in first["runs"] if run["settled"]]
        assert settled and settled[-1]["action_count"] >= 1
        # A live tail is listed and states NO counts.
        for run in first["runs"]:
            if not run["settled"]:
                assert run["action_count"] is None and run["worked_seconds"] is None

        # A scan that cannot land inside the budget is not waited on.
        #
        # The scan is GATED rather than slept: it cannot finish until this test
        # opens the gate, which it does only after the frame has answered. So
        # "the frame answered while its scan was still running" is a fact about
        # ordering that host load cannot change. It replaces `elapsed < 1.0`
        # against a 2 s sleep, which a runner that stalls the request by a
        # second crosses with the product unchanged (docs/TEST_TIME_ASSERTIONS.md).
        # `_SCAN_HANG_S` only bounds a mutant that awaits the scan: it returns
        # once the gate times out, with the scan finished, and fails below.
        ti._reset_for_tests()
        ti.index_path(harness.root, harness.session_id).unlink()
        gate = threading.Event()
        scan_finished = threading.Event()

        def gated_refresh(*_: Any, **__: Any) -> Any:
            gate.wait(_SCAN_HANG_S)
            scan_finished.set()
            return None

        try:
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(ti, "refresh_index", gated_refresh)
                slow = (await harness.client.get(url, params={"limit": 5, "open_frame": 1})).json()[
                    "result"
                ]
                scan_finished_at_answer = scan_finished.is_set()
            assert slow["runs_state"] == "building" and slow["runs"] == []
            # Bounded by the budget, not by the scan and not by the request
            # deadline: the frame answers without its facts rather than making
            # the open pay for them.
            assert not scan_finished_at_answer, "the frame waited for its scan to finish"
        finally:
            gate.set()

        # THE DEADLINE ITSELF, at the seam a slow page read produces: a read whose
        # budget is already spent starts the build (so the frame after it is
        # cheap) and waits for none of it. "None of it" is checked as "the read
        # never suspended": `_frame_index` has no await before its spent-budget
        # return, so any wait at all -- however short -- shows up as a
        # suspension, on a fast host or a loaded one (was `elapsed < 0.5`).
        ti._reset_for_tests()
        gate.clear()
        scan_finished.clear()
        try:
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(ti, "refresh_index", gated_refresh)
                async with harness.pool.session(harness.session_id, read=True) as bridge:
                    index = _finished_without_suspending(
                        bridge._frame_index(time.monotonic() - 1.0)
                    )
            assert index is None
        finally:
            gate.set()


@pytest.mark.asyncio
async def test_a_warm_reopen_does_not_wait_on_its_own_paused_scan(tmp_path: Path) -> None:
    """The re-open of a conversation whose cold scan is still running.

    That scan pauses while this open's paint hold is live (transcript_index "ITS
    OWN OPEN TOO"), so it cannot land inside the snapshot budget and awaiting it
    spends the whole budget idle for the same ``building`` answer -- measured
    112-119 ms of a 120 ms budget on a real ``lop serve``. Mutant: drop the
    ``refresh_paused_for_paint`` guard -> the read waits out the 1 s deadline.
    """
    async with _Harness(tmp_path) as harness:
        session_dir = harness.root / "sessions" / harness.session_id
        rows = _conversation(turns=2)
        rows.append(
            _entry(
                "big",
                1e6,
                "message",
                {"kind": "message", "role": "assistant", "content": [{"text": "x" * (6 << 20)}]},
            )
        )
        _write(session_dir, rows)
        # THE PRECONDITION IS PINNED, not raced. "A scan still running, paused
        # for this open's hold" used to rely on the real 6 MB scan still being
        # in flight and the hold still young (< _YIELD_HOLD_STALE_S, 5 s) when
        # the read arrived; under CPU starvation the bridge open outlasted
        # that, the read (correctly) awaited a no-longer-paused scan, and the
        # test failed 3/3 on `index is None` with the guard intact. The scan is
        # gated so it is in flight by construction, and the hold cannot age.
        scan_gate = threading.Event()
        real_refresh = ti.refresh_index

        def gated_refresh(*args: Any, **kwargs: Any) -> Any:
            scan_gate.wait(_SCAN_HANG_S)
            return real_refresh(*args, **kwargs)

        patch = pytest.MonkeyPatch()
        patch.setattr(ti, "refresh_index", gated_refresh)
        patch.setattr(ti, "_YIELD_HOLD_STALE_S", 600.0)
        try:
            task, started = ti.start_refresh(harness.root, harness.session_id)
            assert started
            release = ti.hold_paint(harness.root, harness.session_id)
            try:
                assert ti.refresh_paused_for_paint(
                    harness.root, harness.session_id
                ), "fixture: the scan must be running and paused for this open"
                async with harness.pool.session(harness.session_id, read=True) as bridge:
                    # A generous deadline, so the only way to answer at once is
                    # to not wait on the paused scan at all -- asserted as
                    # "never suspended" rather than `elapsed < 0.5`, which
                    # measured the host as much as the guard.
                    index = _finished_without_suspending(
                        bridge._frame_index(time.monotonic() + 1.0)
                    )
            finally:
                release()
        finally:
            scan_gate.set()
            patch.undo()
        assert index is None, "answered with facts from a scan paused for this very open"
        assert await asyncio.wait_for(task, 30) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("cache_state", ["large_disk", "small_disk", "large_resident"])
async def test_frame_index_fallback_preserves_ready_facts_without_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cache_state: str
) -> None:
    """No idle budget wager on large disk HITs; resident/small HITs keep facts.

    Observe the actual wait seam rather than asserting milliseconds: host load
    may delay any task, but cannot make waiting on a held refresh useful.
    """
    async with _Harness(tmp_path) as harness:
        harness.seed(turns=4)
        previous = ti.refresh_index(harness.root, harness.session_id)
        assert previous is not None
        cache = ti.index_path(harness.root, harness.session_id)
        # Exercise _frame_index itself, not the earlier summary fast path, and with
        # NO companion published for this publication (nothing calls publish here),
        # so the fallback is the only source — which is the case under test.
        if cache_state != "small_disk":
            with cache.open("a", encoding="utf-8") as handle:
                handle.write(" " * (6 << 20))
        if cache_state == "large_resident":
            ti._remember(ti._key(harness.root, harness.session_id), previous, cache.stat().st_mtime)
        waits = []
        real_wait = asyncio.wait

        async def record_wait(*args: Any, **kwargs: Any) -> Any:
            waits.append(kwargs.get("timeout"))
            return await real_wait(*args, **kwargs)

        release = ti.hold_paint(harness.root, harness.session_id)
        task = None
        try:
            async with harness.pool.session(harness.session_id, read=True) as bridge:
                with monkeypatch.context() as patch:
                    patch.setattr(asyncio, "wait", record_wait)
                    # A GENEROUS budget, because this test observes the wait SEAM
                    # (recorded by the spy above), not milliseconds: on a loaded
                    # host a 0.12 s deadline turned the assertion below into a coin
                    # flip (round-3 M1).
                    index = await bridge._frame_index(time.monotonic() + 5.0)
                entry = ti._IN_FLIGHT.get(ti._key(harness.root, harness.session_id))
                task = entry[1] if entry is not None else None
                if cache_state == "large_disk":
                    assert index is not None, "fallback must not strand its own ready answer"
                else:
                    assert index is not None
                    assert index.runs == previous.runs
                    assert index.checkpoints == previous.checkpoints
                    assert index.journal_rows == previous.journal_rows
                    assert index.source_projection_facts == previous.source_projection_facts
                    assert bool(waits) is (cache_state == "small_disk")
        finally:
            release()
            if task is None:
                entry = ti._IN_FLIGHT.get(ti._key(harness.root, harness.session_id))
                task = entry[1] if entry is not None else None
            if task is not None:
                assert await asyncio.wait_for(task, 30) is not None


@pytest.mark.asyncio
async def test_held_disk_hit_preserves_first_frame_grouping_without_full_hydration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance companion to the scheduling test: fast but incomplete fails.

    These are semantic inputs to collapse/group rendering, not rendered proof.
    A bounded witnessed fact slice must retain them if full hydration is held.
    """
    async with _Harness(tmp_path) as harness:
        harness.seed(turns=4)
        previous = ti.refresh_index(harness.root, harness.session_id)
        assert previous is not None
        _publish(harness.root, harness.session_id, previous)
        cache = ti.index_path(harness.root, harness.session_id)
        with cache.open("a", encoding="utf-8") as handle:
            handle.write(" " * (6 << 20))
        ti._remember(ti._key(harness.root, harness.session_id), previous, cache.stat().st_mtime)
        assert harness.client is not None
        url = f"/v1/desktop/sessions/{harness.session_id}/history"
        params = {"limit": 50, "open_frame": 1}
        control_response = await harness.client.get(url, params=params)
        assert control_response.status_code == 200
        control = control_response.json()["result"]
        assert control["runs_state"] == "ready" and control["runs"]
        ti._reset_for_tests()
        reads = []
        real_read = ti._read_raw

        def read_spy(*args: Any, **kwargs: Any) -> Any:
            reads.append(True)
            return real_read(*args, **kwargs)

        monkeypatch.setattr(ti, "_read_raw", read_spy)
        release = ti.hold_paint(harness.root, harness.session_id)
        try:
            response = await harness.client.get(url, params=params)
            assert response.status_code == 200
            first = response.json()["result"]
            for field in ("entries", "runs", "runs_state", "head_cut"):
                assert (
                    first[field] == control[field]
                ), f"first-frame {field} regressed under paint hold"
            assert reads == [], "grouping still depended on whole-index hydration during paint"
        finally:
            release()
            entry = ti._IN_FLIGHT.get(ti._key(harness.root, harness.session_id))
            if entry is not None:
                await asyncio.wait_for(entry[1], 30)


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["tool", "compacted", "duplicate", "live", "around"])
async def test_projection_preserves_grouped_frames_and_freshness_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str
) -> None:
    """Non-empty folded spans/compaction stay final, not merely rows present."""
    async with _Harness(tmp_path) as harness:
        if shape in {"tool", "live", "duplicate", "around"}:
            rows: list[Any] = []
            for turn in range(1, 4):
                rows.extend(_long_turn(turn, 6))
            if shape == "live":
                rows.extend(_long_turn(9, 8)[:-1])
            directory = harness.root / "sessions" / harness.session_id
            _write(directory, rows)
        else:
            harness.seed(turns=8)
            directory = harness.root / "sessions" / harness.session_id
            # The real compaction path replaces the journal inode.
            await Transcript(directory).compact_file(min_reclaim_bytes=0)
        index = ti.refresh_index(harness.root, harness.session_id)
        assert index is not None
        _publish(harness.root, harness.session_id, index)
        if shape == "duplicate":
            from dataclasses import replace

            ti.write_index(
                harness.root,
                harness.session_id,
                replace(index, checkpoints=index.checkpoints + [index.checkpoints[0]]),
                # A synthetic derivation cannot be published under the same
                # token; the duplicate-id rule is exercised on the canonical
                # document, which the projection must keep agreeing with.
            )
        assert harness.client is not None
        url = f"/v1/desktop/sessions/{harness.session_id}/history"
        params: dict[str, Any] = {
            "limit": 20 if shape == "live" else 50,
            "open_frame": 1,
            "journal_order": 1,
        }
        if shape == "around":
            params = {
                "around_id": "u0003",
                "before": 5,
                "after": 5,
                "open_frame": 1,
                "journal_order": 1,
            }
        control = (await harness.client.get(url, params=params)).json()["result"]
        assert control["runs_state"] == "ready", control
        assert control["journal_order_state"] == "ready", control
        assert any(
            row["payload"].get("tool_calls") or row["payload"].get("role") == "tool"
            for row in control["entries"]
        ) or shape in {"compacted", "around"}
        ti._reset_for_tests()
        reads = []
        original = ti._read_raw

        def spy(*args: Any, **kwargs: Any) -> Any:
            reads.append(True)
            return original(*args, **kwargs)

        monkeypatch.setattr(ti, "_read_raw", spy)
        monkeypatch.setattr(
            DesktopSessionBridge,
            "_frame_index",
            lambda *args, **kwargs: pytest.fail("hydrating fallback ran"),
        )
        release = ti.hold_paint(harness.root, harness.session_id)
        try:
            first = (await harness.client.get(url, params=params)).json()["result"]
            assert first == control
            assert reads == []
        finally:
            release()


@pytest.mark.asyncio
async def test_the_event_stream_states_attaching_only_to_a_negotiating_client(
    tmp_path: Path,
) -> None:
    """C2: the marker is on the FIRST frame, and only for a client that asked.

    ``attaching`` is the fact the snapshot carries one frame later; stating it on
    the ``open`` frame is what lets a renderer start with a neutral working cue
    instead of committing to "nothing is running" and correcting itself in front
    of the reader. A client that did not negotiate the capability gets today's
    payload key for key, which is what makes this additive rather than a change
    of shape.

    Driven through ``bridge.events`` rather than over HTTP, which is this suite's
    convention for stream frames: the generator is the same object the route
    wraps, and it leaves no live ASGI response for a later test to trip on.
    """
    async with _Harness(tmp_path) as harness:
        harness.seed(turns=2)
        async with harness.pool.session(harness.session_id, read=True) as bridge:
            for params, expected in (
                (False, {"subscription_id", "gap", "watch_ttl_seconds"}),
                (True, {"subscription_id", "gap", "watch_ttl_seconds", "attaching"}),
            ):
                subscription = bridge.subscribe()
                stream = bridge.events(subscription, epoch=None, after_seq=0, open_frame=params)
                try:
                    frame = await asyncio.wait_for(stream.__anext__(), timeout=10)
                finally:
                    await stream.aclose()
                assert frame["type"] == "open"
                assert set(frame["payload"]) == expected
                if params:
                    # No owner is attached in this harness, so nothing is coming
                    # and the marker says so: a session with no runtime is never
                    # held back for a dial that does not exist.
                    assert frame["payload"]["attaching"] is False


@pytest.mark.asyncio
async def test_a_snapshot_carries_the_same_page_as_history(tmp_path: Path) -> None:
    """One flag, one shape: the snapshot's embedded page and ``/history`` agree.

    Two unit systems for one page is the drift this capability exists to prevent,
    so the two routes are compared field for field on the same journal.
    """
    async with _Harness(tmp_path) as harness:
        harness.seed(turns=5)
        await harness.warm()
        assert harness.client is not None
        base = f"/v1/desktop/sessions/{harness.session_id}"
        page = (await harness.client.get(f"{base}/history", params={"open_frame": 1})).json()[
            "result"
        ]
        snapshot = (await harness.client.get(base, params={"open_frame": 1})).json()["result"][
            "payload"
        ]["history"]
        assert [row["id"] for row in snapshot["entries"]] == [row["id"] for row in page["entries"]]
        assert snapshot["runs"] == page["runs"]
        assert snapshot["runs_state"] == page["runs_state"]
        assert snapshot["head_cut"] == page["head_cut"]


def _publish(root: Path, sid: str, index) -> None:
    """Write the companion the way the background pass does, synchronously.

    Production publishes it on the event loop after the refresh returns; these
    tests call the derivation directly, so they publish explicitly.
    """
    from local_operator.session import index_read_model as rm

    token = rm.publish(root, sid, index)
    assert token is not None
    rm.write_naming(root, sid, token, index.naming)


def _long_turn(turn: int, calls: int, *, spend_rows: bool = True) -> list[Any]:
    """One turn with ``calls`` tool rows, each followed by a spend receipt.

    THE ORDINARY SHAPE OF A LIVE RUN: ``session.py`` writes a
    ``session_spend.v1`` row per provider round, so a run of N calls holds 2N
    journal rows and one read of ``limit`` rows holds about half that many
    painted ones.
    """
    rows = [
        _entry(
            f"u{turn:04d}",
            float(turn),
            "message",
            {"kind": "message", "role": "user", "content": [{"text": f"turn {turn}"}]},
        )
    ]
    for call in range(calls):
        rows.append(
            _entry(
                f"t{turn:04d}{call:04d}",
                float(turn) + call / 1000,
                "message",
                {
                    "kind": "message",
                    "role": "tool",
                    "content": [{"text": "output"}],
                    "provider_payload": {"details": {}, "duration_s": 1.0},
                },
            )
        )
        if spend_rows:
            rows.append(
                _entry(
                    f"s{turn:04d}{call:04d}",
                    float(turn) + call / 1000 + 0.0005,
                    "custom",
                    {"custom_type": "session_spend.v1", "details": {"micro": 1}},
                )
            )
    rows.append(
        _entry(
            f"a{turn:04d}",
            float(turn) + 1,
            "message",
            {"kind": "message", "role": "assistant", "content": [{"text": "done"}]},
        )
    )
    return rows


@pytest.mark.asyncio
async def test_a_long_live_run_still_serves_the_limit_it_was_asked_for(
    tmp_path: Path,
) -> None:
    """F3(a): the shortcut may not end the read below ``limit`` paintable rows.

    The reachability shortcut exists to stop the EXTRA hunt, not the page: with a
    long live tail (a ``tool`` row and a spend receipt per call) one read of
    ``limit`` raw rows holds about half that many painted ones, and the old rule
    broke out of the loop immediately — S3-class journals answered 50 rows for a
    100-row request.
    """
    async with _Harness(tmp_path) as harness:
        _write(
            harness.root / "sessions" / harness.session_id,
            _long_turn(1, 200),
        )
        assert harness.client is not None
        page = (
            await harness.client.get(
                f"/v1/desktop/sessions/{harness.session_id}/history",
                params={"limit": 100, "open_frame": 1},
            )
        ).json()["result"]
        assert len(page["entries"]) >= 100, len(page["entries"])
        assert page["head_cut"] is True  # the run's head is 400 rows above


@pytest.mark.asyncio
async def test_a_before_id_page_keeps_its_limit_when_the_tail_run_is_long(
    tmp_path: Path,
) -> None:
    """F3(b): an older page is not shortened because the JOURNAL's tail is long.

    The shortcut asked ``tail_head_reachable`` about the journal's last run
    whatever cursor the read carried, so a mid-journal page answered fewer rows
    than asked for whenever the newest turn happened to be a long one.
    """
    async with _Harness(tmp_path) as harness:
        rows: list[Any] = []
        for turn in range(1, 21):
            rows.extend(_long_turn(turn, 3))
        rows.extend(_long_turn(99, 400))
        _write(harness.root / "sessions" / harness.session_id, rows)
        assert harness.client is not None
        page = (
            await harness.client.get(
                f"/v1/desktop/sessions/{harness.session_id}/history",
                params={"before_id": "u0010", "limit": 40, "open_frame": 1},
            )
        ).json()["result"]
        assert len(page["entries"]) >= 40, len(page["entries"])
        # The page ends immediately above the cursor: the newest row before
        # ``u0010`` is the previous turn's answer.
        assert page["entries"][-1]["id"] == "a0009"


@pytest.mark.asyncio
async def test_head_cut_is_true_when_the_cap_drops_the_pages_user_row(
    tmp_path: Path,
) -> None:
    """F2: ``head_cut`` describes what was SERVED, never what was collected.

    A journal the reader could align (its head is a user row) whose page the byte
    or row cap then cut must not claim alignment: the client acts on that flag by
    condensing the page's oldest run from loaded rows.
    """
    async with _Harness(tmp_path) as harness:
        rows = [
            _entry(
                "u0001",
                1.0,
                "message",
                {"kind": "message", "role": "user", "content": [{"text": "go"}]},
            )
        ]
        for call in range(450):
            rows.append(
                _entry(
                    f"t{call:04d}",
                    1.0 + call / 1000,
                    "message",
                    {
                        "kind": "message",
                        "role": "tool",
                        "content": [{"text": "x" * 3000}],
                        "provider_payload": {"details": {}, "duration_s": 1.0},
                    },
                )
            )
        _write(harness.root / "sessions" / harness.session_id, rows)
        assert harness.client is not None
        page = (
            await harness.client.get(
                f"/v1/desktop/sessions/{harness.session_id}/history",
                params={"limit": 500, "open_frame": 1},
            )
        ).json()["result"]
        assert page["head_cut"] is True
        assert page["has_more"] is True
        assert len(page["entries"]) <= 400
        assert all(entry["payload"].get("role") != "user" for entry in page["entries"])


@pytest.mark.asyncio
async def test_a_capped_anchored_page_keeps_the_anchor_and_says_has_more(
    tmp_path: Path,
) -> None:
    """F1: the jump target survives its own page, and the flags stay honest.

    Reproduced against the head before this fix: an anchored page that hit the cap
    dropped the anchor, answered ``has_more: false`` and made the older rows
    unreachable.
    """
    async with _Harness(tmp_path) as harness:
        rows = [
            _entry(
                "u0001",
                1.0,
                "message",
                {"kind": "message", "role": "user", "content": [{"text": "go"}]},
            )
        ]
        for call in range(500):
            rows.append(
                _entry(
                    f"t{call:04d}",
                    1.0 + call / 1000,
                    "message",
                    {
                        "kind": "message",
                        "role": "tool",
                        "content": [{"text": "x" * 3000}],
                        "provider_payload": {"details": {}, "duration_s": 1.0},
                    },
                )
            )
        _write(harness.root / "sessions" / harness.session_id, rows)
        assert harness.client is not None
        page = (
            await harness.client.get(
                f"/v1/desktop/sessions/{harness.session_id}/history",
                params={"around_id": "t0250", "before": 200, "after": 200, "open_frame": 1},
            )
        ).json()["result"]
        ids = [entry["id"] for entry in page["entries"]]
        assert "t0250" in ids, "the anchor vanished from its own page"
        assert page["has_more"] is True
        assert page["head_cut"] is True  # a cap dropped rows from the window
        assert len(ids) <= 400


def test_the_history_page_schema_keeps_its_properties() -> None:
    """F7: the open frame's keys are conditional on the wire AND documented.

    ``exclude_if`` omits a key when it is unset without hiding the model's shape
    from the generated schema, which a ``model_serializer`` returning ``dict``
    does (``additionalProperties: true`` and every property gone).
    """
    schema = HistoryPage.model_json_schema()
    assert set(schema.get("properties", {})) >= {
        "entries",
        "has_more",
        "cursor_missing",
        "has_newer",
        "runs",
        "runs_state",
        "head_cut",
    }
    assert schema.get("title") == "HistoryPage"


@pytest.mark.asyncio
async def test_a_small_limit_still_reaches_the_openings_user_row(tmp_path: Path) -> None:
    """Q3: at ``limit`` <= 20 the hunt ran out of rows it never asked for.

    The reader sizes its window to the REQUEST (it multiplies by a small factor to
    cover rows its own filter drops), so a 5-row page bought a few dozen raw rows
    and the cut stopped short of the run's opening user row — the bar then opened
    on a partial run. The read now buys ``limit + OPEN_FRAME_MAX_EXTRA_ROWS``
    rows, which is the same number the hunt is allowed to spend.
    """
    async with _Harness(tmp_path) as harness:
        harness.seed(turns=8)
        assert harness.client is not None
        base = f"/v1/desktop/sessions/{harness.session_id}/history"
        frame = (await harness.client.get(base, params={"limit": 5, "open_frame": 1})).json()[
            "result"
        ]
        assert len(frame["entries"]) >= 5
        # Either the page reaches a run head, or it says honestly that a cap
        # stopped it — never a headless page reported as complete.
        assert frame["entries"][0]["payload"].get("role") == "user" or frame["head_cut"] is True
        # Facts are NOT asserted here: a page whose bounds hold no run a client
        # could match by answers ``ready`` with an empty list (see ``publish_runs``
        # and the note in ``build_frame``), which is legitimate — this test is
        # about the read's bound, not about the facts.


# ---------------------------------------------------------------------------
# conversation-first-paint R5(ii): the wire rows are built directly, not via a
# dumps+loads round trip, and nothing about them changes.
# ---------------------------------------------------------------------------


def _reference_wire_rows(entries: Any) -> list[dict[str, Any]]:
    """The pre-R5 spelling, verbatim: the oracle for byte identity."""
    import json

    from local_operator.harness.rows import visible_transcript_rows

    return visible_transcript_rows(
        [{**json.loads(entry.to_json()), "ts_source": "entry"} for entry in entries]
    )


def test_wire_rows_are_byte_identical_to_the_round_trip() -> None:
    """Every JSON shape a journal row can hold, plus the two a round trip
    NORMALISES (a tuple, a non-string key) should a page ever carry one: the
    serialized wire must match the old round trip to the byte, key order and
    float spelling included."""
    import json

    entries = [
        _entry("u1", 1.0, "message", {"role": "user", "content": [{"text": "héllo ✓"}]}),
        _entry(
            "a1",
            1.1,
            "message",
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "x"}],
                "usage": {"prompt_tokens": 10, "cost": 0.1 + 0.2, "big": 10**20},
                "flags": [True, False, None, -0.0, 1e-300],
            },
        ),
        _entry("c1", 1.2, "custom", {"custom_type": "x", "details": {"nested": [[{}], []]}}),
        # Separate rows: a non-str key round-trips its WHOLE dict, which would
        # also normalise a tuple beside it and hide that branch.
        _entry("t1", 1.3, "message", {"role": "tool", "content": ("tuple", 1)}),
        _entry("t2", 1.4, "message", {"role": "tool", "meta": {7: "int-key"}}),
    ]
    served = DesktopSessionBridge._wire_rows(entries)
    expected = _reference_wire_rows(entries)
    assert json.dumps(served) == json.dumps(expected)
    assert served == expected


def test_wire_rows_never_alias_the_cached_page() -> None:
    """The page cache SHARES pages between requests, so a wire row a later step
    edits must not reach back into the cached entry (the round trip guaranteed
    that by construction; the direct build must too)."""
    entry = _entry("u1", 1.0, "message", {"role": "user", "content": [{"text": "a"}]})
    row = DesktopSessionBridge._wire_rows([entry])[0]
    row["payload"]["content"][0]["text"] = "edited"
    row["payload"]["role"] = "edited"
    assert entry.payload == {"role": "user", "content": [{"text": "a"}]}


@pytest.mark.asyncio
async def test_an_open_frame_never_aliases_the_cached_page(tmp_path: Path) -> None:
    """openframe-cpu: the frame now strips the page cache's own payloads and
    copies only the rows it serves (``_frame_answer``). A client-side edit of a
    served row — tail or anchored — must still not reach the cached page, or the
    next open would paint the edit. Mutant: drop the copy in ``_frame_answer`` ->
    the second open serves ``"edited"``."""
    async with _Harness(tmp_path) as harness:
        harness.seed(turns=4)
        await harness.warm()
        async with harness.pool.session(harness.session_id, read=True) as bridge:
            first = await bridge.history(limit=6, open_frame=True)
            pristine = json.dumps(first)
            anchor = first["entries"][len(first["entries"]) // 2]["id"]
            around = await bridge.history(
                limit=6, open_frame=True, around_id=anchor, before=2, after=2
            )
            for page in (first, around):
                for entry in page["entries"]:
                    entry["payload"]["edited"] = True
                    for block in entry["payload"].get("content") or []:
                        if isinstance(block, dict):
                            block["text"] = "edited"
            again = await bridge.history(limit=6, open_frame=True)
        assert json.dumps(again) == pristine
        assert "edited" not in json.dumps(again)


def _cached_rows_drifted_from_disk(directory: Path) -> tuple[int, list[str]]:
    """(cached rows checked, ids whose cached payload no longer equals the journal).

    THE JOURNAL ON DISK IS THE REFERENCE, not a snapshot taken mid-test: a
    snapshot taken after the first open would already hold that open's edits and
    compare equal to itself (the first draft of this guard did exactly that).
    """
    from local_operator.session.page_cache import page_cache

    on_disk = {}
    for line in (directory / TRANSCRIPT_FILENAME).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            on_disk[row["id"]] = row["payload"]
    checked, drifted = 0, []
    for _key, (page, _size) in page_cache()._entries.items():
        for entry in page.entries:
            checked += 1
            if entry.payload != on_disk.get(entry.id):
                drifted.append(entry.id)
    return checked, drifted


@pytest.mark.asyncio
async def test_an_open_frame_leaves_the_page_cache_untouched(tmp_path: Path) -> None:
    """THE WINDOW ``StripMemo`` DOCUMENTS (openframe-cpu R2). Between
    ``_wire_rows(copy=False)`` and ``_frame_answer``'s copy, a row's nested
    payload values ARE the page cache's objects, so any in-place edit of a raw
    or stripped row there writes into the cache and reaches the next open. After
    tail, anchored and journal-order opens, every cached row must still equal its
    journal line. Mutant: set ``entry["payload"]["content"][0]["text"] =
    "edited"`` on the served rows inside ``build_frame`` (before the copy) ->
    this test fails."""
    from local_operator.session.page_cache import reset_page_cache

    reset_page_cache()
    async with _Harness(tmp_path) as harness:
        harness.seed(turns=4)
        await harness.warm()
        async with harness.pool.session(harness.session_id, read=True) as bridge:
            first = await bridge.history(limit=6, open_frame=True)
            anchor = first["entries"][len(first["entries"]) // 2]["id"]
            await bridge.history(limit=6, open_frame=True)
            await bridge.history(limit=6, open_frame=True, around_id=anchor, before=2, after=2)
            await bridge.history(limit=6, open_frame=True, journal_order=True, order_ids=[])
        checked, drifted = _cached_rows_drifted_from_disk(
            harness.root / "sessions" / harness.session_id
        )
    reset_page_cache()
    assert checked > 0  # the opens went through the cache
    assert drifted == []


@pytest.mark.asyncio
async def test_an_incrementally_refreshed_index_serves_the_full_rebuilds_frame(
    tmp_path: Path,
) -> None:
    """O1 (#3034 QA): the open frame may not depend on HOW the index was refreshed.

    ``tail_head_reachable`` subtracts the tail run's ``first_seq`` from
    ``scan.rows``. An incremental refresh used to record only the rows its suffix
    re-read there, so the subtraction went to zero and the shortcut kept hunting
    for a head the full scan's count correctly called out of reach. The tail run
    below is that shape: its head is more raw rows above the end than the hunt
    may cover, but few enough PAINTABLE rows that a hunt left running reaches it.
    """
    async with _Harness(tmp_path) as harness:
        rows: list[Any] = []
        for turn in range(1, 41):
            rows.extend(_long_turn(turn, 3))
        rows.extend(_long_turn(99, 100))
        rows.pop()  # the tail run is still LIVE: no closing answer yet
        directory = harness.root / "sessions" / harness.session_id
        _write(directory, rows)
        await harness.warm()
        # One more tool row through the real writer: the next refresh is incremental.
        await Transcript(directory).append_message(
            Message(role="tool", content=[TextContent(text="output")], tool_call_id="c-extra")
        )
        await harness.warm()
        assert harness.client is not None
        url = f"/v1/desktop/sessions/{harness.session_id}/history"
        params = {"limit": 50, "open_frame": 1}
        incremental = (await harness.client.get(url, params=params)).json()["result"]
        ti._reset_for_tests()
        ti.index_path(harness.root, harness.session_id).unlink()
        await harness.warm()
        full = (await harness.client.get(url, params=params)).json()["result"]

        def canonical(page: dict[str, Any]) -> dict[str, Any]:
            return {k: v for k, v in page.items() if k not in {"epoch", "journal_generation"}}

        assert full["head_cut"] is True  # the shape under test: the head is out of reach
        assert canonical(incremental) == canonical(full)


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [20, 100])
@pytest.mark.parametrize("case", ["one", "fifty", "new_run"])
async def test_a_frame_whose_index_lands_during_the_hunt_is_the_steady_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str, limit: int
) -> None:
    """The desktop twin of the relay's pre-index cell (#3037 QA, O1).

    ``_frame_index`` at attempt 0 can answer ``None`` (the refresh did not land
    inside the open budget) and the final call after the hunt can answer a
    resident index (it landed meanwhile). The frame then carried the hunt's rows
    under ``ready`` — rows the steady read refuses because the index calls the
    tail run's head out of reach. Held still here: the first call answers None,
    every later one is the real call.
    """
    from local_operator.server.utils.desktop_sessions import DesktopSessionBridge

    async with _Harness(tmp_path) as harness:
        rows: list[Any] = []
        for turn in range(1, 3):
            rows.extend(_long_turn(turn, 2))
        rows.extend(_long_turn(99, 110))
        rows.pop()  # the tail run is still LIVE: no closing answer yet
        directory = harness.root / "sessions" / harness.session_id
        _write(directory, rows)
        await harness.warm()
        transcript = Transcript(directory)
        if case == "new_run":
            await transcript.append_message(
                Message(role="assistant", content=[TextContent(text="done")])
            )
            await transcript.append_message(
                Message(role="user", content=[TextContent(text="next")])
            )
        else:
            for call in range(1 if case == "one" else 50):
                await transcript.append_message(
                    Message(
                        role="tool", content=[TextContent(text="output")], tool_call_id=f"x{call}"
                    )
                )
        await harness.warm()
        assert harness.client is not None
        url = f"/v1/desktop/sessions/{harness.session_id}/history"
        params = {"limit": limit, "open_frame": 1}

        original = DesktopSessionBridge._frame_index
        # The RESIDENT entry must stay: it is what makes the retry cheap enough
        # to land inside the hunt, and the case under test is a retry that
        # SUCCEEDS. Establishing it HERE rather than inheriting whatever
        # ``warm()`` happened to leave makes the precondition explicit — a
        # competition-era read can serve the manifest without remembering the
        # resident, and this test then failed for a reason it does not measure
        # (round-3 M1). Only the on-disk document (and so the companion) goes, so
        # the first attempt has nothing and must answer None.
        index = ti.refresh_index(harness.root, harness.session_id)
        assert index is not None
        key = ti._key(harness.root, harness.session_id)
        ti._remember(key, index, ti.index_path(harness.root, harness.session_id).stat().st_mtime)
        ti.index_path(harness.root, harness.session_id).unlink()
        calls: list[int] = []

        async def _lands_during_the_hunt(self: Any, deadline: float) -> Any:
            calls.append(1)
            if len(calls) == 1:
                return None
            # LAND the index inside this call instead of betting on a background
            # refresh fitting the open budget: the case under test is "the index
            # arrives during the hunt", and the arrival must be deterministic
            # (round-3 M1: the old shape failed or passed on scheduling luck).
            arrival = await asyncio.to_thread(ti.refresh_index, harness.root, harness.session_id)
            if arrival is not None:
                ti._remember(
                    key,
                    arrival,
                    ti.index_path(harness.root, harness.session_id).stat().st_mtime,
                )
            return await original(self, deadline)

        monkeypatch.setattr(DesktopSessionBridge, "_frame_index", _lands_during_the_hunt)
        first = (await harness.client.get(url, params=params)).json()["result"]
        assert len(calls) >= 2, "the read never re-asked for the index"
        monkeypatch.setattr(DesktopSessionBridge, "_frame_index", original)
        steady = (await harness.client.get(url, params=params)).json()["result"]

        def canonical(page: dict[str, Any]) -> dict[str, Any]:
            return {
                k: v for k, v in page.items() if k not in {"epoch", "journal_generation", "cwd"}
            }

        assert steady["runs_state"] == "ready" and first["runs_state"] == "ready"
        assert (len(first["entries"]), first["head_cut"]) == (
            len(steady["entries"]),
            steady["head_cut"],
        )
        assert canonical(first) == canonical(steady)


@pytest.mark.asyncio
async def test_projection_journal_ordinals_match_the_resident_join(tmp_path: Path) -> None:
    """R2/Q1: coordinates describe what the page SERVES, never a dropped row.

    Same code, same fixture, same request: the resident join is the control, and
    the projection must produce its payload exactly. The pre-fix projection added
    the opener of a row the cap had dropped (``limit=5`` answered ``{u0005}``
    where the head served ``{u0001..u0005}``).
    """
    async with _Harness(tmp_path) as harness:
        harness.seed(turns=6)
        index = ti.refresh_index(harness.root, harness.session_id)
        assert index is not None
        _publish(harness.root, harness.session_id, index)
        key = ti._key(harness.root, harness.session_id)
        ti._remember(key, index, ti.index_path(harness.root, harness.session_id).stat().st_mtime)
        assert harness.client is not None
        url = f"/v1/desktop/sessions/{harness.session_id}/history"
        params = {"limit": 5, "open_frame": 1, "journal_order": 1}
        control = (await harness.client.get(url, params=params)).json()["result"]
        assert control["journal_order_state"] == "ready"
        # The resident join emits an opener axis, not one entry per served row:
        # each row carries its own ``journal_seq``.
        assert control["journal_ordinals"], control
        assert all(row.get("journal_seq") is not None for row in control["entries"])

        ti._reset_for_tests()  # no resident index: the companion is the only source
        projected = (await harness.client.get(url, params=params)).json()["result"]
        assert projected == control


@pytest.mark.asyncio
async def test_a_warm_checkpoint_manifest_reflects_a_later_append(tmp_path: Path) -> None:
    """Round-3 N1: the rail must not answer the PRE-append list forever.

    Warmed first on purpose: the cached manifest is keyed on the canonical
    document and the naming sidecar, and an append moves neither file, so a
    cache that consults only those two keeps answering `ready N` while the
    journal has grown (QA's "append loses facts", base answers `ready N+2`).
    """
    async with _Harness(tmp_path) as harness:
        harness.seed(turns=3)
        index = ti.refresh_index(harness.root, harness.session_id)
        assert index is not None
        _publish(harness.root, harness.session_id, index)
        assert harness.client is not None
        url = f"/v1/desktop/sessions/{harness.session_id}/checkpoints"
        warm = (await harness.client.get(url)).json()["result"]
        assert warm["index"]["state"] == "ready"
        before = len(warm["checkpoints"])
        directory = harness.root / "sessions" / harness.session_id
        await Transcript(directory).append_message(
            Message(role="user", content=[TextContent(text="after the warm cache")])
        )
        await Transcript(directory).append_message(
            Message(role="assistant", content=[TextContent(text="answered")])
        )
        after = (await harness.client.get(url)).json()["result"]
        polled = [after]
        for _ in range(2):
            polled.append((await harness.client.get(url)).json()["result"])
        assert all(poll["index"]["state"] == "ready" for poll in polled)
        assert all(len(poll["checkpoints"]) == before + 2 for poll in polled), [
            len(poll["checkpoints"]) for poll in polled
        ]
        assert len({len(poll["checkpoints"]) for poll in polled}) == 1


@pytest.mark.asyncio
async def test_an_append_warm_reopen_still_reaches_final_runs(tmp_path: Path) -> None:
    """Q5: the companion must not cost the append path its ready first frame.

    A companion published for the PREVIOUS journal is stale for the suffix, so the
    read falls back to the refresh the ORIGINAL path waits on; that wait is what
    used to land inside the frame budget (5/6 quiet opens) and what a lock-held
    companion build pushed outside it (0/6). Deterministic here: after an append
    the first frame must carry the settled runs the steady frame carries, not a
    ``building`` placeholder.
    """
    async with _Harness(tmp_path) as harness:
        harness.seed(turns=3)
        ti.refresh_index(harness.root, harness.session_id)
        directory = harness.root / "sessions" / harness.session_id
        await Transcript(directory).append_message(
            Message(role="tool", content=[TextContent(text="appended")], tool_call_id="x1")
        )
        ti._reset_for_tests()
        assert harness.client is not None
        url = f"/v1/desktop/sessions/{harness.session_id}/history"
        params = {"limit": 50, "open_frame": 1}
        first = (await harness.client.get(url, params=params)).json()["result"]
        steady = (await harness.client.get(url, params=params)).json()["result"]
        assert steady["runs_state"] == "ready"
        if first["runs_state"] == "ready":
            # The discriminator: a READY first frame must carry the appended runs,
            # never the pre-append set a stale companion would have served.
            assert first["runs"] == steady["runs"], "a ready frame carried stale runs"
        else:
            # An honest building answer is allowed under host load, but ONLY with
            # no facts — an incomplete ready frame is the forbidden case, and this
            # branch used to compare itself against steady unconditionally, which
            # made the test fail for the very answer it permits (round-3 M1).
            assert first["runs"] == [], "a building frame carried facts"


# ---------------------------------------------------------------------------
# journal_order=1 ON ITS OWN: the flag a renderer older than the open frame sends
# ---------------------------------------------------------------------------


def _order_params(ids: list[str], limit: int = 50) -> dict[str, Any]:
    """The page a client asks for when it declares ONLY journal order."""
    return {"limit": limit, "journal_order": 1, "order_ids": ids}


@pytest.mark.asyncio
async def test_a_warm_plain_journal_order_first_frame_is_ready(tmp_path: Path) -> None:
    """THE FIRST PAINT, FOR THE FLAG THAT HAS NO ``open_frame`` TWIN.

    ``journal_order=1`` is its own capability, and a renderer that declares it
    without ``open_frame=1`` reaches a page built with no runs and no bounds. That
    read is answered from the read-model companion now; before it was answered
    from a resident index alone, so a process that had just started said
    ``building`` and reconciled later — the after-paint change the capability
    exists to remove. The resident join is the control: same store, same request,
    and the two payloads must be identical, opener closure included.
    """
    async with _Harness(tmp_path) as harness:
        harness.seed(turns=6)
        index = ti.refresh_index(harness.root, harness.session_id)
        assert index is not None
        _publish(harness.root, harness.session_id, index)
        ids = ["ans05", "user05"]
        ti._reset_for_tests()  # a fresh process: the companion is the only source
        assert harness.client is not None
        url = f"/v1/desktop/sessions/{harness.session_id}/history"
        projected = (await harness.client.get(url, params=_order_params(ids))).json()["result"]
        assert projected["journal_order_state"] == "ready", projected.get("journal_order_state")
        assert all(row.get("journal_seq") is not None for row in projected["entries"])
        assert projected["journal_ordinals"], projected
        assert projected["journal_generation"]

        ti._remember(
            ti._key(harness.root, harness.session_id),
            index,
            ti.index_path(harness.root, harness.session_id).stat().st_mtime,
        )
        control = (await harness.client.get(url, params=_order_params(ids))).json()["result"]
        assert control["journal_order_state"] == "ready"
        assert projected == control, "the companion's page differs from the resident join"


@pytest.mark.asyncio
async def test_a_cold_plain_journal_order_still_answers_building(tmp_path: Path) -> None:
    """Nothing published: the answer is today's, not an invented ``ready``."""
    async with _Harness(tmp_path) as harness:
        harness.seed(turns=6)
        ti._reset_for_tests()
        assert harness.client is not None
        url = f"/v1/desktop/sessions/{harness.session_id}/history"
        first = (await harness.client.get(url, params=_order_params(["ans05"]))).json()["result"]
        assert first["journal_order_state"] == "building"
        assert not first.get("journal_ordinals")
        assert not first.get("journal_generation")
        assert all(row.get("journal_seq") is None for row in first["entries"])


@pytest.mark.asyncio
async def test_a_missing_companion_plain_journal_order_degrades_to_building(
    tmp_path: Path,
) -> None:
    """An older or swept companion is not an error and never a fabricated fact."""
    async with _Harness(tmp_path) as harness:
        harness.seed(turns=6)
        index = ti.refresh_index(harness.root, harness.session_id)
        assert index is not None
        _publish(harness.root, harness.session_id, index)
        index_read_model.header(harness.root, harness.session_id).value.path.unlink()
        ti._reset_for_tests()
        assert harness.client is not None
        url = f"/v1/desktop/sessions/{harness.session_id}/history"
        first = (await harness.client.get(url, params=_order_params(["ans05"]))).json()["result"]
        assert first["journal_order_state"] == "building"
        assert not first.get("journal_ordinals")


@pytest.mark.asyncio
async def test_an_append_after_publish_yields_the_appended_rows_coordinates(
    tmp_path: Path,
) -> None:
    """THE APPEND PATH IS THE RISKY ONE, so it is pinned through the real route.

    A companion published for the PREVIOUS journal is stale for the suffix. The
    read must not answer ``ready`` from it, and it must not answer a coordinate
    for a row that projection never derived: the first frame either carries the
    appended rows' ordinals (the refresh landed inside the budget) or says
    ``building`` with none of them — and the settled frame then carries them all.
    """
    async with _Harness(tmp_path) as harness:
        harness.seed(turns=3)
        index = ti.refresh_index(harness.root, harness.session_id)
        assert index is not None
        _publish(harness.root, harness.session_id, index)
        directory = harness.root / "sessions" / harness.session_id
        await Transcript(directory).append_message(
            Message(role="user", content=[TextContent(text="after the publication")])
        )
        await Transcript(directory).append_message(
            Message(role="assistant", content=[TextContent(text="answered")])
        )
        ti._reset_for_tests()
        assert harness.client is not None
        url = f"/v1/desktop/sessions/{harness.session_id}/history"
        ids = ["ans02", "user02"]
        first = (await harness.client.get(url, params=_order_params(ids))).json()["result"]
        # The refresh the first read started is what carries the appended rows;
        # poll it in rather than assuming one round trip is enough. ``steady``
        # starts as the answer already in hand so the assertion below reads a
        # value even if the loop's own body never runs.
        steady: dict[str, Any] = first
        for _ in range(200):
            steady = (await harness.client.get(url, params=_order_params(ids))).json()["result"]
            if steady["journal_order_state"] == "ready":
                break
            await asyncio.sleep(0.05)
        assert steady["journal_order_state"] == "ready"
        assert all(row.get("journal_seq") is not None for row in steady["entries"])
        if first["journal_order_state"] == "ready":
            assert first["entries"] == steady["entries"], "a ready frame served stale rows"
        else:
            assert first["journal_order_state"] == "building"
            assert not first.get("journal_ordinals"), "a building frame carried coordinates"


@pytest.mark.asyncio
async def test_a_companion_answer_arms_the_resident_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The projection must not become the only source the process ever uses.

    "THE SUITE CANNOT TELL THE FIX FROM ITS ABSENCE" is the failure this test
    exists for: delete the warm and every assertion below still holds except the
    two that matter. A page answered from the companion never enters the
    ``building`` branch, and that branch was the only thing that started a scan —
    so without the arm the poll loop reopens the companion forever (measured
    25/25 opens on a 36.5 MB journal, against 1/25 once armed).

    COUNTED, NOT TIMED: the number of companion opens is the mechanism, and it is
    the one number host load cannot move. ``fresh_resident`` is asserted directly
    for the same reason.
    """
    opens: list[str] = []
    original = index_read_model.order_facts

    def counted(*args: Any, **kwargs: Any) -> Any:
        opens.append(str(kwargs.get("order_ids") or args))
        return original(*args, **kwargs)

    monkeypatch.setattr(index_read_model, "order_facts", counted)
    async with _Harness(tmp_path) as harness:
        harness.seed(turns=6)
        index = ti.refresh_index(harness.root, harness.session_id)
        assert index is not None
        _publish(harness.root, harness.session_id, index)
        ti._reset_for_tests()  # a fresh process: the companion is the only source
        assert harness.client is not None
        url = f"/v1/desktop/sessions/{harness.session_id}/history"
        params = _order_params(["ans05", "user05"])
        first = (await harness.client.get(url, params=params)).json()["result"]
        assert first["journal_order_state"] == "ready"
        assert opens, "the companion did not serve the first frame at all"
        for _ in range(200):
            if ti.fresh_resident(harness.root, harness.session_id) is not None:
                break
            await asyncio.sleep(0.02)
        assert (
            ti.fresh_resident(harness.root, harness.session_id) is not None
        ), "a companion answer did not arm the resident scan, so every poll reopens it"
        after_arm = len(opens)
        poll = (await harness.client.get(url, params=params)).json()["result"]
        assert poll["journal_order_state"] == "ready"
        assert poll["journal_ordinals"] == first["journal_ordinals"]
        assert (
            len(opens) == after_arm
        ), "a poll reopened the companion although the resident index was in hand"


@pytest.mark.asyncio
async def test_a_failing_resident_scan_is_retrieved_and_never_reaches_the_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The armed scan is not awaited, so its failure must still be observed.

    ``start_refresh``'s wrapper logs and RE-RAISES, and this caller owns no await
    on the task: without a done-callback the exception surfaces later as
    ``Task exception was never retrieved`` at GC — a second, misleading traceback
    after the page that armed it (review round 2, MAJOR-1). The page must be
    unaffected either way, which is the other half of the assertion.
    """
    loop = asyncio.get_running_loop()
    unretrieved: list[dict[str, Any]] = []
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: unretrieved.append(context))

    async def boom() -> Any:
        raise RuntimeError("the scan refused to run")

    def failing(*_args: Any, **_kwargs: Any) -> Any:
        return asyncio.ensure_future(boom()), True

    monkeypatch.setattr(ti, "fresh_resident", lambda *_a, **_k: None)
    monkeypatch.setattr(ti, "start_refresh", failing)
    try:
        async with _Harness(tmp_path) as harness:
            harness.seed(turns=6)
            index = ti.refresh_index(harness.root, harness.session_id)
            assert index is not None
            _publish(harness.root, harness.session_id, index)
            ti._reset_for_tests()
            assert harness.client is not None
            url = f"/v1/desktop/sessions/{harness.session_id}/history"
            params = _order_params(["ans05"])
            page = (await harness.client.get(url, params=params)).json()["result"]
            assert page["journal_order_state"] == "ready"
            # Let the armed task finish and die, then look for the leak.
            for _ in range(20):
                await asyncio.sleep(0.01)
                gc.collect()
            assert not unretrieved, f"the armed scan's failure leaked: {unretrieved}"
    finally:
        loop.set_exception_handler(previous)


@pytest.mark.asyncio
async def test_a_slow_companion_does_not_hold_the_snapshots_first_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The journal-order companion read is bounded by the paint deadline.

    REGRESSION (operator's v0.106.7 desktop, 2026-11-17): on a 743 MB
    journal ``a.hist.journal_order`` measured 1,547 and 3,081 ms on two sidebar
    opens, because ``_journal_order_page`` awaited ``order_facts`` with no bound
    while every other open-frame hop was capped by #3904. The snapshot must
    answer ``building`` inside its budget, and the existing correction must then
    land the order as a ``journal_order`` event (what the renderer reconciles
    from), with the resident scan armed so the next read needs no companion.

    The control arm pins the other half of the contract: a companion that
    answers inside the deadline still serves a READY first frame. Its budget is
    a hang ceiling (as in ``test_the_first_frame_spends_only_what_is_left``), not
    a speed claim about the runner.
    """
    from local_operator.server.utils import desktop_sessions as desktop_sessions_utils
    from local_operator.session.runtime.types import JOURNAL_ORDER_EVENT

    # Steer the frame onto the page path: no summary, so no ``frame_facts`` can
    # answer the order first and the companion is the hop under test.
    async def no_frame_source(self: Any, *_a: Any, **_k: Any) -> Any:
        return None

    monkeypatch.setattr(DesktopSessionBridge, "_frame_source", no_frame_source)
    published: list[dict[str, Any]] = []
    original_publish = DesktopSessionBridge.publish

    def recording(self: Any, kind: str, payload: dict[str, Any], **kwargs: Any) -> None:
        if kind == "event" and payload.get("type") == JOURNAL_ORDER_EVENT:
            published.append(payload)
        original_publish(self, kind, payload, **kwargs)

    monkeypatch.setattr(DesktopSessionBridge, "publish", recording)
    original_facts = index_read_model.order_facts
    async with _Harness(tmp_path) as harness:
        harness.seed(turns=6)
        index = ti.refresh_index(harness.root, harness.session_id)
        assert index is not None
        _publish(harness.root, harness.session_id, index)
        assert harness.client is not None
        url = f"/v1/desktop/sessions/{harness.session_id}"
        params = {"open_frame": 1, "limit": 50, "journal_order": 1, "order_ids": ["ans05"]}

        # CONTROL: a companion inside the deadline is used.
        ti._reset_for_tests()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(desktop_sessions_utils, "OPEN_FRAME_SNAPSHOT_BUDGET_S", 600.0)
            ready = (await harness.client.get(url, params=params)).json()["result"]["payload"]
        assert ready["history"]["journal_order_state"] == "ready"

        # THE REGRESSION: a companion that outlives the deadline is not waited on.
        # The control armed a resident scan; let it land before forgetting it,
        # or it would publish after the reset and answer the next read itself.
        for _ in range(500):
            if ti.fresh_resident(harness.root, harness.session_id) is not None:
                break
            await asyncio.sleep(0.01)
        ti._reset_for_tests()
        bridge = harness.pool.bridges[harness.session_id]
        assert ti.fresh_resident(harness.root, harness.session_id) is None
        published.clear()
        released = threading.Event()

        def slow_facts(*args: Any, **kwargs: Any) -> Any:
            released.wait(5.0)
            return original_facts(*args, **kwargs)

        monkeypatch.setattr(index_read_model, "order_facts", slow_facts)
        try:
            started = time.monotonic()
            slow = (await harness.client.get(url, params=params)).json()["result"]["payload"]
            elapsed = time.monotonic() - started
        finally:
            released.set()
        assert slow["history"]["journal_order_state"] == "building"
        assert elapsed < 1.0, f"the snapshot waited {elapsed:.2f}s on the companion"
        # The correction lands the order after first paint, and arms the resident.
        correction = bridge._order_correction
        assert correction is not None
        await asyncio.wait_for(asyncio.shield(correction), timeout=30)
        rows = [row for frame in published for row in frame.get("rows", ())]
        assert any(
            row["id"] == "ans05" and row.get("journal_seq") is not None for row in rows
        ), published
        assert ti.fresh_resident(harness.root, harness.session_id) is not None


# ---------------------------------------------------------------------------
# Large journals: the companion answers, serve never hydrates (ADR step 2,
# docs/design/transcript-index-off-process.md). "Large" is
# ``ti.COMPANION_FIRST_MIN_BYTES``; these tests lower it so a six-turn fixture
# qualifies, because the behaviour under test is the branch, not the bytes.
# ---------------------------------------------------------------------------


def _count_hydrations(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every whole-index build or hydration serve performs, by session.

    ``refresh_index`` is the one door ``start_refresh`` (and therefore every
    open-path warm, correction and fallback) goes through to hydrate a persisted
    index or scan a journal, so counting its calls counts hydrations.
    """
    calls: list[str] = []
    original = ti.refresh_index

    def counted(config_dir: Any, session_id: str) -> Any:
        calls.append(session_id)
        return original(config_dir, session_id)

    monkeypatch.setattr(ti, "refresh_index", counted)
    return calls


def _append_raw(directory: Path, row: dict[str, Any]) -> None:
    """Append one journal row in place (same inode), as a live runtime does."""
    with (directory / TRANSCRIPT_FILENAME).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")


@pytest.mark.asyncio
async def test_switching_across_six_large_sessions_never_hydrates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Six published large journals, each opened twice: zero ``refresh_index``.

    On main a companion answer armed ``_warm_resident_scan`` -- a full hydration
    of the persisted index on serve's GIL -- and the 4-slot resident LRU evicted
    it again once the operator moved across more than four conversations, so
    the hydration was re-paid on most switches. Counted, not timed.
    """
    monkeypatch.setattr(ti, "COMPANION_FIRST_MIN_BYTES", 1)
    async with _Harness(tmp_path) as harness:
        sids = [harness.session_id] + [
            await harness.pool.create(str(harness.root)) for _ in range(5)
        ]
        for sid in sids:
            _write(harness.root / "sessions" / sid, _conversation(6))
            index = ti.refresh_index(harness.root, sid)
            assert index is not None
            _publish(harness.root, sid, index)
        ti._reset_for_tests()  # a fresh serve: only the companions exist
        calls = _count_hydrations(monkeypatch)
        assert harness.client is not None
        for _round in range(2):
            for sid in sids:
                url = f"/v1/desktop/sessions/{sid}/history"
                for params in (
                    _order_params(["ans05", "user05"]),
                    {"limit": 5, "open_frame": 1, "journal_order": 1},
                ):
                    body = (await harness.client.get(url, params=params)).json()["result"]
                    assert body["journal_order_state"] == "ready", (sid, params)
                await asyncio.sleep(0)
        # Let any armed background task run before counting.
        for _ in range(10):
            await asyncio.sleep(0.01)
        assert calls == [], f"serve hydrated large journals on open: {calls}"
        assert all(ti.fresh_resident(harness.root, sid) is None for sid in sids)


@pytest.mark.asyncio
async def test_a_large_journal_companion_answer_is_the_resident_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Byte-equal to the resident join for the same journal, both read shapes."""
    monkeypatch.setattr(ti, "COMPANION_FIRST_MIN_BYTES", 1)
    async with _Harness(tmp_path) as harness:
        harness.seed(turns=6)
        index = ti.refresh_index(harness.root, harness.session_id)
        assert index is not None
        _publish(harness.root, harness.session_id, index)
        assert harness.client is not None
        url = f"/v1/desktop/sessions/{harness.session_id}/history"
        for params in (
            _order_params(["ans05", "user05", "user01"]),
            {"limit": 5, "open_frame": 1, "journal_order": 1},
        ):
            ti._reset_for_tests()
            projected = (await harness.client.get(url, params=params)).json()["result"]
            ti._remember(
                ti._key(harness.root, harness.session_id),
                index,
                ti.index_path(harness.root, harness.session_id).stat().st_mtime,
            )
            control = (await harness.client.get(url, params=params)).json()["result"]
            assert control["journal_order_state"] == "ready"
            assert json.dumps(projected, sort_keys=True) == json.dumps(control, sort_keys=True)


@pytest.mark.asyncio
async def test_a_grown_large_journal_answers_its_prefix_from_the_companion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE APPEND-PREFIX RULE: a journal grown by run-neutral rows stays ``ready``.

    A live session appends bookkeeping rows (attention, spend, frontend state
    checkpoints) constantly, which moved the journal's signature and made the
    companion ``stale``: main answered ``building`` and hydrated serve's whole
    index to correct it. The appended rows are read directly and given their
    ordinals; every answer must equal the resident index of the GROWN journal,
    for both read shapes, with zero hydrations on the way.
    """
    monkeypatch.setattr(ti, "COMPANION_FIRST_MIN_BYTES", 1)
    async with _Harness(tmp_path) as harness:
        harness.seed(turns=6)
        index = ti.refresh_index(harness.root, harness.session_id)
        assert index is not None
        _publish(harness.root, harness.session_id, index)
        directory = harness.root / "sessions" / harness.session_id
        for id_, custom_type, details in (
            # No ``attention_started``: a start row can unsettle the tail run, so
            # it is not answer-neutral (review round 1, M1) and falls back.
            ("name-after", "conversation_name", {"name": "renamed"}),
            ("spend-after", "session_spend.v1", {"usd": 0.01}),
            ("bk-after", "frontend_state_checkpoint_v1", {"blob": "x" * 4096}),
        ):
            _append_raw(
                directory,
                {
                    "id": id_,
                    "ts": time.time(),
                    "type": "custom",
                    "payload": {"custom_type": custom_type, "details": details},
                },
            )
        shapes = (
            _order_params(["ans05", "user05", "user02", "bk-after"]),
            {"limit": 5, "open_frame": 1, "journal_order": 1, "order_ids": ["user02"]},
        )
        ti._reset_for_tests()
        calls = _count_hydrations(monkeypatch)
        assert harness.client is not None
        url = f"/v1/desktop/sessions/{harness.session_id}/history"
        projected = []
        for params in shapes:
            body = (await harness.client.get(url, params=params)).json()["result"]
            assert body["journal_order_state"] == "ready", (params, body.get("journal_order_state"))
            projected.append(body)
        assert "bk-after" in {row["id"] for row in projected[0]["entries"]}
        assert calls == [], "a run-neutral append still hydrated"

        grown = ti.refresh_index(harness.root, harness.session_id)
        assert grown is not None and grown.sig["size"] > index.sig["size"]
        ti._remember(
            ti._key(harness.root, harness.session_id),
            grown,
            ti.index_path(harness.root, harness.session_id).stat().st_mtime,
        )
        for params, body in zip(shapes, projected):
            control = (await harness.client.get(url, params=params)).json()["result"]
            assert control["journal_order_state"] == "ready"
            assert json.dumps(body, sort_keys=True) == json.dumps(control, sort_keys=True), params


@pytest.mark.asyncio
async def test_a_grown_large_journal_never_serves_a_stale_ready_for_new_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Visible rows past the prefix: ``building`` first, then every coordinate.

    The prefix rule must not answer for a row the companion never derived; the
    page that reaches one falls to the existing correction, which settles it.
    """
    monkeypatch.setattr(ti, "COMPANION_FIRST_MIN_BYTES", 1)
    async with _Harness(tmp_path) as harness:
        harness.seed(turns=3)
        index = ti.refresh_index(harness.root, harness.session_id)
        assert index is not None
        _publish(harness.root, harness.session_id, index)
        directory = harness.root / "sessions" / harness.session_id
        await Transcript(directory).append_message(
            Message(role="user", content=[TextContent(text="after the publication")])
        )
        ti._reset_for_tests()
        assert harness.client is not None
        url = f"/v1/desktop/sessions/{harness.session_id}/history"
        params = _order_params(["ans02", "user02"])
        first = (await harness.client.get(url, params=params)).json()["result"]
        assert first["journal_order_state"] == "building"
        assert not first.get("journal_ordinals"), "a building frame carried coordinates"
        steady: dict[str, Any] = first
        for _ in range(200):
            steady = (await harness.client.get(url, params=params)).json()["result"]
            if steady["journal_order_state"] == "ready":
                break
            await asyncio.sleep(0.05)
        assert steady["journal_order_state"] == "ready"
        assert all(row.get("journal_seq") is not None for row in steady["entries"])


@pytest.mark.asyncio
async def test_a_small_journal_keeps_arming_the_resident_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Below the threshold nothing changed: a companion answer still warms."""
    monkeypatch.setattr(ti, "COMPANION_FIRST_MIN_BYTES", 1 << 40)
    async with _Harness(tmp_path) as harness:
        harness.seed(turns=6)
        index = ti.refresh_index(harness.root, harness.session_id)
        assert index is not None
        _publish(harness.root, harness.session_id, index)
        ti._reset_for_tests()
        calls = _count_hydrations(monkeypatch)
        assert harness.client is not None
        url = f"/v1/desktop/sessions/{harness.session_id}/history"
        first = (await harness.client.get(url, params=_order_params(["ans05"]))).json()["result"]
        assert first["journal_order_state"] == "ready"
        for _ in range(200):
            if ti.fresh_resident(harness.root, harness.session_id) is not None:
                break
            await asyncio.sleep(0.02)
        assert calls == [harness.session_id]


async def _corrections_after_a_deadline_open(
    harness: "_Harness", monkeypatch: pytest.MonkeyPatch, params: dict[str, Any]
) -> list[dict[str, Any]]:
    """Open with every companion read past the paint deadline; return the events.

    This is the loaded-host shape the real-path rig measured: the 120 ms budget
    is gone before the first companion read, the frame says ``building``, and
    the after-paint ``_order_correction`` publishes the coordinates.
    """
    from local_operator.server.utils import desktop_sessions as module
    from local_operator.session.runtime.types import JOURNAL_ORDER_EVENT

    events: list[dict[str, Any]] = []
    original_publish = module.DesktopSessionBridge.publish
    original_bounded = module._bounded_frame_read

    def publish(self: Any, kind: str, payload: dict[str, Any], **kwargs: Any) -> None:
        if isinstance(payload, dict) and payload.get("type") == JOURNAL_ORDER_EVENT:
            events.append(payload)
        original_publish(self, kind, payload, **kwargs)

    async def past_deadline(deadline: Any, func: Any, /, *args: Any, **kwargs: Any) -> Any:
        if deadline is not None:
            return False, None
        return await original_bounded(deadline, func, *args, **kwargs)

    monkeypatch.setattr(module.DesktopSessionBridge, "publish", publish)
    monkeypatch.setattr(module, "_bounded_frame_read", past_deadline)
    assert harness.client is not None
    url = f"/v1/desktop/sessions/{harness.session_id}"
    first = (await harness.client.get(url, params=params)).json()["result"]
    history = first["payload"]["history"]
    assert history.get("journal_order_state") == "building", history.get("journal_order_state")
    for _ in range(200):
        if events:
            break
        await asyncio.sleep(0.05)
    monkeypatch.setattr(module.DesktopSessionBridge, "publish", original_publish)
    monkeypatch.setattr(module, "_bounded_frame_read", original_bounded)
    return events


@pytest.mark.asyncio
async def test_a_large_journal_correction_comes_from_the_companion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The after-paint correction publishes the resident path's events, unhydrated.

    Main answered every deadline-missed open of a large journal by hydrating the
    whole index (``start_refresh``) to publish these coordinates. Same journal,
    same request: the companion arm's rows must equal the resident arm's, and
    the companion arm must not call ``refresh_index`` at all.
    """
    params = {"open_frame": 1, "journal_order": 1, "limit": 5, "order_ids": ["user02"]}
    async with _Harness(tmp_path) as harness:
        harness.seed(turns=6)
        index = ti.refresh_index(harness.root, harness.session_id)
        assert index is not None
        _publish(harness.root, harness.session_id, index)
        ti._reset_for_tests()

        monkeypatch.setattr(ti, "COMPANION_FIRST_MIN_BYTES", 1)
        calls = _count_hydrations(monkeypatch)
        companion = await _corrections_after_a_deadline_open(harness, monkeypatch, params)
        assert companion and all(event.get("rows") for event in companion), companion
        assert calls == [], "the correction hydrated a large journal"
        assert {event["journal_generation"] for event in companion} == {index.journal_generation}

        monkeypatch.setattr(ti, "COMPANION_FIRST_MIN_BYTES", 1 << 40)
        resident = await _corrections_after_a_deadline_open(harness, monkeypatch, params)
        assert calls, "the control arm never took the resident path"
        assert [event["rows"] for event in companion] == [event["rows"] for event in resident]
        assert {event["journal_generation"] for event in resident} == {index.journal_generation}


@pytest.mark.asyncio
async def test_rows_queued_while_the_companion_correction_awaits_are_answered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review round 1, m1: a page that queues rows mid-correction is not stranded.

    While ``_companion_order_correction`` awaits its worker read, a page that
    falls to ``_journal_page`` adds pending rows but starts no task of its own
    (the correction is not done). The first version consumed only its snapshot
    and returned ``True``, so those rows got no event until some later page
    started another correction.
    """
    from local_operator.server.utils import desktop_sessions as module
    from local_operator.session.runtime.types import JOURNAL_ORDER_EVENT

    async with _Harness(tmp_path) as harness:
        harness.seed(turns=2)
        source = (1, 2, 3.0)
        async with harness.pool.session(harness.session_id, read=True) as bridge:
            bridge._order_pending = {"user01": ({"id": "user01"}, source)}
            reads: list[list[str]] = []

            def order_selection(*_: Any, order_ids: list[str], **__: Any) -> Any:
                reads.append(list(order_ids))
                if len(reads) == 1:
                    # The concurrent page, landing while the first read is out.
                    bridge._order_pending["ans01"] = ({"id": "ans01"}, source)
                return index_read_model.Result(
                    "ready",
                    index_read_model.FrameSelection(
                        checkpoints=[],
                        runs=[],
                        messages=[],
                        journal_rows={"user01": 0, "ans01": 1},
                        journal_generation="gen",
                    ),
                )

            events: list[dict[str, Any]] = []
            monkeypatch.setattr(module.index_read_model, "order_selection", order_selection)
            monkeypatch.setattr(
                module.transcript_index,
                "journal_order_correction_batches",
                lambda _selection, rows: [rows] if rows else [],
            )
            monkeypatch.setattr(
                bridge,
                "publish",
                lambda kind, payload, **_: (
                    events.append(payload) if payload.get("type") == JOURNAL_ORDER_EVENT else None
                ),
            )
            assert await bridge._companion_order_correction() is True
            assert reads == [["user01"], ["ans01"]]
            published = [row["id"] for event in events for row in event["rows"]]
            assert published == ["user01", "ans01"]
            assert bridge._order_pending == {}
