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
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.server.routes import desktop_sessions
from local_operator.server.utils.desktop_sessions import DesktopSessions
from local_operator.session import transcript_index as ti
from local_operator.session.transcript import TRANSCRIPT_FILENAME, TranscriptEntry

TOKEN = "synthetic-desktop-token"


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
    """
    async with _Harness(tmp_path) as harness:
        harness.seed(turns=4)
        assert harness.client is not None
        url = f"/v1/desktop/sessions/{harness.session_id}/history"

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
        ti._reset_for_tests()

        def slow_refresh(*_: Any, **__: Any) -> Any:
            time.sleep(2.0)
            return None

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(ti, "refresh_index", slow_refresh)
            started = time.monotonic()
            slow = (await harness.client.get(url, params={"limit": 5, "open_frame": 1})).json()[
                "result"
            ]
            elapsed = time.monotonic() - started
        assert slow["runs_state"] == "building" and slow["runs"] == []
        # Bounded by the budget, not by the scan (2 s) and not by the request
        # deadline: the frame answers without its facts rather than making the
        # open pay for them.
        assert elapsed < 1.0

        # THE DEADLINE ITSELF, at the seam a slow page read produces: a read whose
        # budget is already spent starts the build (so the frame after it is
        # cheap) and waits for none of it.
        ti._reset_for_tests()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(ti, "refresh_index", slow_refresh)
            async with harness.pool.session(harness.session_id, read=True) as bridge:
                started = time.monotonic()
                index = await bridge._frame_index(time.monotonic() - 1.0)
                elapsed_deadline = time.monotonic() - started
        assert index is None
        assert elapsed_deadline < 0.5


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
