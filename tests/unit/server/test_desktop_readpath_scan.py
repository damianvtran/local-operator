"""The desktop open read decodes a page, never the whole journal.

WOULD-HAVE-FAILED-BEFORE. ``DesktopSessionBridge._with_peer_cycle_records``
built a ``Transcript`` for every snapshot and every ``/history`` page purely to
pass its ``directory`` to ``discover_records``; ``Transcript.__init__`` eagerly
JSON-decodes the complete journal. On the operator's Aida session that is 485 MB
/ 70,353 rows and a measured 6.6-8.9 s per read, synchronously on the event
loop; the index refresh that followed cost another 1.5-2.5 s per request.

The invariant asserted here is ROWS DECODED, not milliseconds: one test counts
``TranscriptEntry.from_json`` calls around the join itself and bounds them to
the page; the route-level tests count ``Transcript`` constructions and index
lookups across a real HTTP request; and one bridge-level test guards that the
warm door still serves proof. A wall-clock threshold would fail on a loaded host
without saying anything about the defect; a row count fails on exactly the
regression this file exists for (the old code decoded every row in the seeded
journal).

Row counts are also why the fixtures are a few thousand small rows rather than a
half-gigabyte journal: the invariant is about rows DECODED, not bytes.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.harness.peer_cycle_display import record_id
from local_operator.server.routes import desktop_sessions
from local_operator.server.utils.desktop_sessions import DesktopSessions
from local_operator.session import transcript_index as ti
from local_operator.session.transcript import (
    ENTRY_MESSAGE,
    TRANSCRIPT_FILENAME,
    Transcript,
    TranscriptEntry,
)

TOKEN = "synthetic-desktop-token"

#: Rows the old construction replayed on the very first read. Big enough that the
#: old whole-journal decode is unmistakable, small enough to seed in milliseconds.
JOURNAL_ROWS = 4000

#: A page read is allowed to decode its own window plus cursor/continuation
#: lookahead rows. ``limit=100`` therefore sits far below this bound, while the
#: regression (one decode per journal row) sits far above it.
PAGE_DECODE_BOUND = 500


@pytest.fixture(autouse=True)
def _desktop_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))


def _journal(rows: int, *, session_id: str | None = None) -> list[TranscriptEntry]:
    """A plain conversation whose LAST row is a page-carrying ``eval`` root.

    THE TAIL ROW IS THE TRIGGER. ``_with_peer_cycle_records`` derives candidates
    from the selected page; a journal with no ``eval``/``no_reply`` assistant call
    returns before touching any index, which would hide the very regression this
    file pins. The tail page therefore carries exactly one such assistant row.
    """
    entries = [
        TranscriptEntry(
            f"u{index:05d}",
            100.0 + index,
            ENTRY_MESSAGE,
            {"kind": "message", "role": "user", "content": [{"text": f"question {index}"}]},
        )
        for index in range(rows)
    ]
    # A COMPACTION ANCHOR AND A FRONTEND CHECKPOINT, because every real journal
    # carries them and the cold-replay reader stops only when the newest
    # compaction AND its ``first_kept_entry_id`` row, plus the newest required
    # custom row, are all in the buffer. Without both it walks to BOF -- which is
    # the correct fallback for a journal that has neither, and would make this
    # fixture measure cold-replay cost instead of the peer-cycle join it pins.
    entries.append(
        TranscriptEntry(
            "cmp-tail",
            100.0 + rows - 0.75,
            "compaction",
            {
                "summary": "earlier turns condensed",
                "first_kept_entry_id": f"u{rows - 1:05d}",
                "tokens_before": 1000,
            },
        )
    )
    entries.append(
        TranscriptEntry(
            "ckpt-tail",
            100.0 + rows - 0.5,
            "custom",
            {
                "custom_type": "frontend_state_checkpoint_v1",
                "details": {"state": {"epoch": "e0", "session_id": "seed", "sequence": rows}},
            },
        )
    )
    if session_id is not None:
        # A DECOY ARCHIVE ROW under the candidate's own deterministic id. It is
        # not a display record (wrong custom_type), so the join must report it
        # ``unavailable`` -- which is how this test observes that the resident
        # index REALLY served the archive read rather than the join being skipped.
        entries.append(
            TranscriptEntry(
                record_id(session_id, "a-last", "root-call"),
                100.0 + rows - 0.25,
                "custom",
                {"custom_type": "not_a_display_record", "details": {}},
            )
        )
    entries.append(
        TranscriptEntry(
            "a-last",
            100.0 + rows,
            ENTRY_MESSAGE,
            {
                "kind": "message",
                "role": "assistant",
                "content": [{"text": "working"}],
                "tool_calls": [{"id": "root-call", "name": "eval", "arguments": {"code": "op"}}],
            },
        )
    )
    return entries


def _write(directory: Path, rows: list[TranscriptEntry]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    lines = [row.to_json() for row in rows]
    (directory / TRANSCRIPT_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


class _Harness:
    """The real router over a synthetic store, as the other desktop tests run it."""

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

    def seed(self, rows: int = JOURNAL_ROWS, *, decoy: bool = False) -> None:
        _write(
            self.root / "sessions" / self.session_id,
            _journal(rows, session_id=self.session_id if decoy else None),
        )

    async def warm(self) -> None:
        """Publish a CURRENT resident index, the way the rail's poll does."""
        task, _started = ti.start_refresh(self.root, self.session_id)
        assert await task is not None


def _count_decodes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count ``TranscriptEntry.from_json`` calls from here on (one list entry each)."""
    calls: list[int] = []
    original = TranscriptEntry.from_json

    def counting(line: str) -> TranscriptEntry | None:
        calls.append(1)
        return original(line)

    monkeypatch.setattr(TranscriptEntry, "from_json", staticmethod(counting))
    return calls


@pytest.mark.asyncio
async def test_the_join_decodes_only_the_rows_it_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The join over a 4,003-row journal decodes a bounded number of rows.

    This isolates the regression at the method that owned it: the join used to
    build ``Transcript``, so one call decoded every seeded row (70,353 on the
    operator's store). Cold replay and the index build are deliberately OUTSIDE
    the counted window -- a first open legitimately pays those once -- so this
    stays a statement about the join and about rows decoded, never milliseconds.
    """
    from local_operator.session.transcript import read_transcript_page

    async with _Harness(tmp_path) as harness:
        harness.seed()
        await harness.warm()
        page = read_transcript_page(tmp_path / "sessions" / harness.session_id, limit=100)
        answer = {
            "entries": [json.loads(entry.to_json()) for entry in page.entries],
            "has_more": page.has_more,
            "cursor_missing": page.reconciled,
            "has_newer": page.has_newer,
        }
        # The bridge's own cold acquisition is a ONE-TIME cost (it may replay for
        # the attach) and is not what this file measures; take it before counting.
        async with harness.pool.session(harness.session_id) as bridge:
            decoded = _count_decodes(monkeypatch)
            joined = await bridge._with_peer_cycle_records(answer, [page])
        assert joined["entries"][-1]["id"] == "a-last"
        assert len(decoded) <= PAGE_DECODE_BOUND, (
            f"the peer-cycle join decoded {len(decoded)} journal rows for a 100-row "
            f"page over a {JOURNAL_ROWS + 1}-row journal; it replayed the whole "
            "journal instead of using the directory and a bounded archive read"
        )


@pytest.mark.asyncio
async def test_a_repeat_open_never_constructs_a_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repeat read must not replay the journal through ``Transcript``.

    ``Transcript.__init__`` JSON-decodes the whole file; the join used it only to
    read ``.directory``, and did so on EVERY snapshot and EVERY page. The first
    request is warmed (the pool's cold acquisition pays its own one-time attach),
    then the same request is measured: a construction here is the join's.
    """
    from local_operator.session.transcript import Transcript

    async with _Harness(tmp_path) as harness:
        harness.seed()
        url = f"/v1/desktop/sessions/{harness.session_id}/history"
        assert harness.client is not None
        warm = await harness.client.get(url, params={"limit": 100})
        assert warm.status_code == 200, warm.text
        constructions: list[Any] = []
        original = Transcript.__init__

        def counting(self: Any, *args: Any, **kwargs: Any) -> None:
            # Placeholder only: see the note in the repeat-open test -- reading
            # ``self.directory`` before the original raises AttributeError on the
            # very code whose behaviour this asserts about.
            constructions.append(None)
            original(self, *args, **kwargs)

        monkeypatch.setattr(Transcript, "__init__", counting)
        assert harness.client is not None
        response = await harness.client.get(url, params={"limit": 100})
        assert response.status_code == 200, response.text
        assert response.json()["result"]["entries"][-1]["id"] == "a-last"
        assert constructions == [], (
            "a repeat desktop read constructed Transcript, replaying the whole "
            f"journal ({len(constructions)} time(s)); the join needs only the "
            "journal directory"
        )


@pytest.mark.asyncio
async def test_a_page_without_peer_roots_reads_no_journal_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ordinary page owes no proof, so the join must not look anything up.

    This is the zero-cost door: candidates come from the page's own messages, and
    with none there is no index read, no archive read and no replay. The index is
    left COLD on purpose -- a warm one would make ``fresh_resident`` cheap enough
    to hide a lookup that should never happen.
    """
    async with _Harness(tmp_path) as harness:
        harness.seed()
        # Replace the eval row with a plain assistant turn: nothing to prove.
        directory = tmp_path / "sessions" / harness.session_id
        rows = _journal(JOURNAL_ROWS)
        rows[-1] = TranscriptEntry(
            "a-last",
            100.0 + JOURNAL_ROWS,
            ENTRY_MESSAGE,
            {"kind": "message", "role": "assistant", "content": [{"text": "done"}]},
        )
        _write(directory, rows)
        ti._reset_for_tests()
        lookups: list[str] = []
        constructions: list[Any] = []
        original_init = Transcript.__init__

        def counting_init(self: Any, *args: Any, **kwargs: Any) -> None:
            # A placeholder, NOT ``self.directory``: that attribute is assigned
            # by ``__init__`` itself, so reading it first would turn the
            # pre-change failure into an AttributeError instead of the invariant
            # this test asserts.
            constructions.append(None)
            original_init(self, *args, **kwargs)

        monkeypatch.setattr(Transcript, "__init__", counting_init)
        monkeypatch.setattr(ti, "start_refresh", lambda *args, **kwargs: lookups.append("refresh"))
        monkeypatch.setattr(
            ti, "fresh_resident", lambda *args, **kwargs: lookups.append("resident")
        )
        assert harness.client is not None
        response = await harness.client.get(f"/v1/desktop/sessions/{harness.session_id}/history")
        assert response.status_code == 200, response.text
        assert response.json()["result"]["entries"][-1]["id"] == "a-last"
        assert lookups == [], f"a page with no eval/no_reply roots touched the index: {lookups}"
        assert constructions == [], (
            "a page with no eval/no_reply roots replayed the journal; the join must "
            "return before touching the transcript or the index"
        )


@pytest.mark.asyncio
async def test_without_a_resident_index_the_read_builds_and_answers_no_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cold index starts the existing single-flight build, never a sync scan.

    The request answers with the EXPLICIT empty partition naming this page's
    candidates ("no proof yet", distinguishable from the key's absence, which
    means the page owes no proof) and leaves the build to ``start_refresh``
    (which parses in a thread and publishes the resident index for the next
    read). ``refresh_index`` on the request path is the regression -- 1.5-2.5 s
    of cache parsing on the event loop -- so it is armed here to fail loudly if
    anything reaches for it.
    """
    async with _Harness(tmp_path) as harness:
        harness.seed(rows=400)
        ti._reset_for_tests()
        started: list[tuple[Path, str]] = []

        def fake_start(config_dir: str | Path, session_id: str) -> tuple[asyncio.Task[None], bool]:
            started.append((Path(config_dir), session_id))
            # A REAL task, so the done-callback the bridge installs to retrieve a
            # failed build's exception is actually exercised (a ``None`` return
            # would be swallowed by the caller's broad except).
            return asyncio.create_task(asyncio.sleep(0)), True

        def forbidden_refresh(*_: Any, **__: Any) -> Any:
            raise AssertionError("refresh_index must not run on the desktop read path")

        monkeypatch.setattr(ti, "start_refresh", fake_start)
        monkeypatch.setattr(ti, "refresh_index", forbidden_refresh)
        # The BRIDGE's answer: the HTTP response model (``HistoryPage``) does not
        # carry this renderer-only key, and this test is about the join.
        async with harness.pool.session(harness.session_id) as bridge:
            page = await bridge.history(limit=50)
        assert page["entries"][-1]["id"] == "a-last"
        expected = record_id(harness.session_id, "a-last", "root-call")
        assert page["peer_cycle_display_records"] == {
            "records": [],
            "remaining": [],
            "unavailable": [expected],
            "bytes": 0,
        }, "a cold index must answer the explicit empty partition, not omit the key"
        assert started == [
            (tmp_path, harness.session_id)
        ], "the cold read did not start the single-flight index build exactly once"


@pytest.mark.asyncio
async def test_the_bridge_history_serves_proof_once_the_resident_index_is_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fast door is still a DOOR: a current index serves the join.

    Bounding the read is not permission to drop proof. With the resident index
    warm, the next bridge read takes the archive path: the seeded decoy row
    carries the candidate's own id but is not a display record, so it must come
    back ``unavailable`` -- never silently omitted -- and its presence in the
    answer proves the index (not a journal replay) supplied the join.
    """
    async with _Harness(tmp_path) as harness:
        harness.seed(rows=400, decoy=True)
        await harness.warm()
        expected = record_id(harness.session_id, "a-last", "root-call")
        # Read the BRIDGE's answer: the HTTP response model (``HistoryPage``)
        # does not carry this renderer-only key, and this test is about the join,
        # not about that existing wire contract.
        async with harness.pool.session(harness.session_id) as bridge:
            page = await bridge.history(limit=100)
        records = page.get("peer_cycle_display_records")
        assert records is not None, "a warm resident index did not serve the proof join"
        assert records["records"] == []
        assert records["unavailable"] == [expected]
