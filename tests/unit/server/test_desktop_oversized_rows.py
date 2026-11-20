"""Oversized rows on the opening reads, and the ref route that fetches them whole.

THE DEFECT (measured on main b3edd7ba through the real serve path): a newest
row that is ONE 12,583,147 B tool message made a 12,594,749 B opening
snapshot, over the desktop renderer's 8 MiB per-record cap, so the
conversation could not be opened; the bare ``/history`` page had no byte bound
at all, and a 10.59 MB ``frontend_state_checkpoint_v1`` row made a 12.13 MB
snapshot. These tests pin the fix at the level a client meets it — the real
routes over a synthetic HOME — and each one FAILS on main's code (the size
asserts, or the missing ``truncated`` marker the route tests start from).

The UI cap is restated here as a literal on purpose: it lives in the other
repository, and the test must not pass because a core constant drifted.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.server.routes import desktop_sessions
from local_operator.server.utils import desktop_sessions as module
from local_operator.server.utils.desktop_sessions import DesktopSessions
from local_operator.session import transcript_index as ti
from local_operator.session.transcript import TRANSCRIPT_FILENAME, TranscriptEntry

TOKEN = "synthetic-desktop-token"
#: The desktop renderer's per-record IPC cap (local-operator-ui).
UI_RECORD_CAP = 8_388_608
#: The open frame's page budget, restated for the same reason.
PAGE_BUDGET = 1_572_864
#: The measured C4 row size.
C4_ROW_BYTES = 12_583_147


@pytest.fixture(autouse=True)
def _env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    for name in [key for key in os.environ if key.startswith("CMUX_")]:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _fresh_snapshot_pool() -> Any:
    module.shutdown_snapshot_pool(wait=True)
    yield
    module.shutdown_snapshot_pool(wait=True)


def _message(id_: str, ts: float, role: str, text: str) -> TranscriptEntry:
    return TranscriptEntry(
        id_, ts, "message", {"kind": "message", "role": role, "content": [{"text": text}]}
    )


def _sized_tool_row(id_: str, ts: float, total: int) -> TranscriptEntry:
    """A tool message whose journal line is exactly ``total`` bytes."""
    row = _message(id_, ts, "tool", "")
    pad = total - len(row.to_json().encode())
    row = _message(id_, ts, "tool", "".join(chr(ord("a") + n % 26) for n in range(pad)))
    assert len(row.to_json().encode()) == total
    return row


def _checkpoint(id_: str, ts: float, jobs: int) -> TranscriptEntry:
    """A ``frontend_state_checkpoint_v1`` row the size of a 4k-job roster (~10.6 MB)."""
    roster = [{"id": f"job{n:05d}", "summary": "s" * 2600} for n in range(jobs)]
    return TranscriptEntry(
        id_,
        ts,
        "custom",
        {
            "custom_type": "frontend_state_checkpoint_v1",
            "details": {"state": {"session_id": "seed", "jobs": roster}},
        },
    )


def _opening(n: int = 3) -> list[TranscriptEntry]:
    rows: list[TranscriptEntry] = []
    for turn in range(n):
        rows.append(_message(f"user{turn}", 100.0 + 2 * turn, "user", f"turn {turn}"))
        rows.append(_message(f"ans{turn}", 101.0 + 2 * turn, "assistant", f"answer {turn}"))
    return rows


class _Harness:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.app = FastAPI()
        self.app.include_router(desktop_sessions.router)
        self.pool = DesktopSessions(root)
        self.app.state.desktop_sessions = self.pool
        self.app.state.config_manager = ConfigManager(config_dir=root)
        self.session_id = ""
        self.client: AsyncClient

    async def __aenter__(self) -> "_Harness":
        self.session_id = await self.pool.create(str(self.root))
        self.client = AsyncClient(
            transport=ASGITransport(app=self.app),
            base_url="http://localhost",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.client.aclose()
        ti._reset_for_tests()

    def seed(self, rows: list[TranscriptEntry], session_id: str | None = None) -> None:
        directory = self.root / "sessions" / (session_id or self.session_id)
        directory.mkdir(parents=True, exist_ok=True)
        text = "\n".join(row.to_json() for row in rows) + "\n"
        (directory / TRANSCRIPT_FILENAME).write_text(text, encoding="utf-8")

    @property
    def base(self) -> str:
        return f"/v1/desktop/sessions/{self.session_id}"


def _newest(page: dict[str, Any]) -> dict[str, Any]:
    return page["entries"][-1]


@pytest.mark.asyncio
@pytest.mark.parametrize("open_frame", [1, 0])
async def test_c4_a_newest_oversized_row_is_clipped_under_the_cap(
    tmp_path: Path, open_frame: int
) -> None:
    """C4: one 12.58 MB newest tool row; the snapshot fits and the row is present."""
    async with _Harness(tmp_path) as harness:
        harness.seed([*_opening(), _sized_tool_row("huge", 200.0, C4_ROW_BYTES)])
        response = await harness.client.get(harness.base, params={"open_frame": open_frame})
        assert response.status_code == 200
        assert len(response.content) <= UI_RECORD_CAP
        history = response.json()["result"]["payload"]["history"]
        newest = _newest(history)
        assert newest["id"] == "huge", "the newest row is still admitted"
        marker = newest["truncated"]
        assert marker["reason"] == ("open_frame_cap" if open_frame else "history_cap")
        assert marker["bytes_total"] > C4_ROW_BYTES  # wire row = journal line + ts_source
        assert marker["shown_bytes"] <= PAGE_BUDGET
        assert isinstance(marker["ref"], str) and marker["ref"]
        assert "omitted" not in marker
        # The projection keeps the row's shape and the start of its text, and
        # spends the budget on it: a clip that kept only the envelope would
        # also "fit" (the first draft did exactly that — 135 bytes shown).
        assert newest["payload"]["content"][0]["text"].startswith("abcdef")
        assert marker["shown_bytes"] >= PAGE_BUDGET * 0.9
        assert len(newest["payload"]["content"][0]["text"]) >= PAGE_BUDGET * 0.9


@pytest.mark.asyncio
@pytest.mark.parametrize("open_frame", [1, 0])
async def test_c4b_a_short_reply_after_the_big_row_needs_no_marker_on_the_reply(
    tmp_path: Path, open_frame: int
) -> None:
    """C4b: the reply is newest; nothing on the page exceeds the cap."""
    async with _Harness(tmp_path) as harness:
        harness.seed(
            [
                *_opening(),
                _sized_tool_row("huge", 200.0, C4_ROW_BYTES),
                _message("reply", 201.0, "assistant", "done"),
            ]
        )
        response = await harness.client.get(harness.base, params={"open_frame": open_frame})
        assert len(response.content) <= UI_RECORD_CAP
        history = response.json()["result"]["payload"]["history"]
        assert _newest(history)["id"] == "reply"
        assert "truncated" not in _newest(history)
        # No row anywhere on the page is served over the page budget.
        for row in history["entries"]:
            assert len(json.dumps(row).encode()) <= PAGE_BUDGET + 512


@pytest.mark.asyncio
async def test_bare_history_withholds_an_oversized_checkpoint_and_bounds_the_page(
    tmp_path: Path,
) -> None:
    """Bare ``/history``: the 10.6 MB checkpoint is withheld, the page is bounded."""
    async with _Harness(tmp_path) as harness:
        checkpoint = _checkpoint("ckpt", 300.0, 4000)
        assert len(checkpoint.to_json().encode()) > 10_000_000
        harness.seed([*_opening(), _sized_tool_row("big", 200.0, 3_000_000), checkpoint])
        snapshot = await harness.client.get(harness.base)
        assert len(snapshot.content) <= UI_RECORD_CAP
        page_response = await harness.client.get(f"{harness.base}/history")
        assert len(page_response.content) <= UI_RECORD_CAP
        page = page_response.json()["result"]
        newest = _newest(page)
        assert newest["id"] == "ckpt"
        assert newest["payload"] == {"custom_type": "frontend_state_checkpoint_v1"}
        assert newest["truncated"]["omitted"] is True
        assert newest["truncated"]["reason"] == "history_cap"
        big = next(row for row in page["entries"] if row["id"] == "big")
        assert big["truncated"]["reason"] == "history_cap"


@pytest.mark.asyncio
async def test_bare_history_total_bound_leaves_older_rows_to_the_next_page(
    tmp_path: Path,
) -> None:
    """Five 1.4 MB rows exceed the 4 MiB page: the OLDEST go, ``has_more`` says so."""
    async with _Harness(tmp_path) as harness:
        rows = [_sized_tool_row(f"t{n}", 200.0 + n, 1_400_000) for n in range(5)]
        harness.seed([*_opening(1), *rows])
        page = (await harness.client.get(f"{harness.base}/history")).json()["result"]
        ids = [row["id"] for row in page["entries"]]
        assert ids[-1] == "t4" and "t0" not in ids
        assert page["has_more"] is True
        assert sum(len(json.dumps(row).encode()) for row in page["entries"]) <= 4 * 1024 * 1024
        older = (
            await harness.client.get(f"{harness.base}/history", params={"before_id": ids[0]})
        ).json()["result"]
        assert older["entries"][-1]["id"] == f"t{int(ids[0][1:]) - 1}"


@pytest.mark.asyncio
async def test_the_in_band_resync_snapshot_goes_through_the_projection(tmp_path: Path) -> None:
    """An oversized live frame resyncs; the resync snapshot is clipped like an open."""
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    directory = tmp_path / "sessions" / sid
    directory.mkdir(parents=True, exist_ok=True)
    rows = [*_opening(), _sized_tool_row("huge", 200.0, C4_ROW_BYTES)]
    (directory / TRANSCRIPT_FILENAME).write_text(
        "\n".join(row.to_json() for row in rows) + "\n", encoding="utf-8"
    )
    async with pool.session(sid) as bridge:
        sub = bridge.subscribe()
        stream = bridge.events(sub, epoch=bridge.epoch, after_seq=bridge.sequence)
        assert (await anext(stream))["type"] == "open"
        assert (await anext(stream))["type"] == "snapshot"
        huge = {"type": "history_delta", "reset": True, "messages": [{"t": "x" * 9_000_000}]}
        bridge.publish("event", huge)
        assert sub.resync_pending
        gap = await anext(stream)
        resynced = await anext(stream)
        assert gap["type"] == "gap" and resynced["type"] == "snapshot"
        assert len(json.dumps(resynced).encode()) <= UI_RECORD_CAP
        newest = resynced["payload"]["history"]["entries"][-1]
        assert newest["id"] == "huge"
        assert newest["truncated"]["reason"] == "history_cap"
        await stream.aclose()


async def _fetch_all(harness: _Harness, ref: str, step: int) -> tuple[bytes, list[int]]:
    data = b""
    offset = 0
    limits: list[int] = []
    while True:
        response = await harness.client.get(
            f"{harness.base}/history/rows/{ref}", params={"offset": offset, "limit": step}
        )
        assert response.status_code == 200, response.text
        body = response.json()["result"]
        limits.append(body["limit"])
        assert body["offset"] == offset
        chunk = base64.b64decode(body["data"])
        assert len(chunk) == body["length"]
        data += chunk
        offset += len(chunk)
        if body["eof"]:
            return data, limits


@pytest.mark.asyncio
@pytest.mark.parametrize("open_frame", [1, 0])
async def test_the_rows_route_reassembles_the_full_row_byte_for_byte(
    tmp_path: Path, open_frame: int
) -> None:
    """Happy path, and an oversized ``limit`` is CLAMPED to 1 MiB, not refused."""
    async with _Harness(tmp_path) as harness:
        huge = _sized_tool_row("huge", 200.0, C4_ROW_BYTES)
        harness.seed([*_opening(), huge])
        page = (
            await harness.client.get(f"{harness.base}/history", params={"open_frame": open_frame})
        ).json()["result"]
        marker = _newest(page)["truncated"]
        data, limits = await _fetch_all(harness, marker["ref"], step=50_000_000)
        assert set(limits) == {1_048_576}, "an oversized limit is clamped"
        assert len(data) == marker["bytes_total"]
        expected = {**json.loads(huge.to_json()), "ts_source": "entry"}
        assert json.loads(data.decode("utf-8")) == expected
        assert data == json.dumps(expected, separators=(",", ":"), ensure_ascii=False).encode()
        # A small, uneven step reassembles to the same bytes.
        small, _ = await _fetch_all(harness, marker["ref"], step=999_983)
        assert small == data


@pytest.mark.asyncio
async def test_the_rows_route_refuses_what_history_refuses_and_foreign_refs(
    tmp_path: Path,
) -> None:
    """401 without the token; wrong session, foreign and forged refs are 404; 416 past end."""
    async with _Harness(tmp_path) as harness:
        harness.seed([*_opening(), _sized_tool_row("huge", 200.0, C4_ROW_BYTES)])
        ref = _newest((await harness.client.get(f"{harness.base}/history")).json()["result"])[
            "truncated"
        ]["ref"]
        route = f"{harness.base}/history/rows/{ref}"

        async with AsyncClient(
            transport=ASGITransport(app=harness.app), base_url="http://localhost"
        ) as anonymous:
            assert (await anonymous.get(route)).status_code == 401
            assert (await anonymous.get(f"{harness.base}/history")).status_code == 401

        # The same row id in ANOTHER session: that session's own ref works, ours does not.
        other = await harness.pool.create(str(tmp_path))
        harness.seed([*_opening(), _sized_tool_row("huge", 200.0, C4_ROW_BYTES)], other)
        foreign = f"/v1/desktop/sessions/{other}/history/rows/{ref}"
        assert (await harness.client.get(foreign)).status_code == 404
        other_ref = _newest(
            (await harness.client.get(f"/v1/desktop/sessions/{other}/history")).json()["result"]
        )["truncated"]["ref"]
        assert other_ref != ref, "refs are not guessable across sessions"
        crossed = await harness.client.get(f"{harness.base}/history/rows/{other_ref}")
        assert crossed.status_code == 404
        assert (
            await harness.client.get(f"/v1/desktop/sessions/{other}/history/rows/{other_ref}")
        ).status_code == 200

        # A session that does not exist answers what /history answers for it.
        missing = "/v1/desktop/sessions/does-not-exist"
        absent_row = await harness.client.get(f"{missing}/history/rows/{ref}")
        absent_page = await harness.client.get(f"{missing}/history")
        assert absent_row.status_code == absent_page.status_code

        # A raw row id, or a ref with its tag swapped, is not a ref.
        prefix, encoded_id, _tag = ref.split(".")
        forged = f"{prefix}.{encoded_id}.{'A' * 22}"
        for bad in ("huge", forged, "r1.not-base64!.x"):
            refused = await harness.client.get(f"{harness.base}/history/rows/{bad}")
            assert refused.status_code == 404

        page = (await harness.client.get(f"{harness.base}/history")).json()["result"]
        total = _newest(page)["truncated"]["bytes_total"]
        at_end = await harness.client.get(route, params={"offset": total})
        assert at_end.status_code == 200 and at_end.json()["result"]["eof"] is True
        assert at_end.json()["result"]["length"] == 0
        past = await harness.client.get(route, params={"offset": total + 1})
        assert past.status_code == 416
        assert (await harness.client.get(route, params={"offset": -1})).status_code == 422
        assert (await harness.client.get(route, params={"limit": 0})).status_code == 422


def _text_row(id_: str, ts: float, char: str, count: int) -> TranscriptEntry:
    return _message(id_, ts, "tool", char * count)


@pytest.mark.asyncio
@pytest.mark.parametrize("char", ["ж", "三", "😀"])
async def test_a_non_latin_snapshot_fits_the_cap_on_the_sse_wire(tmp_path: Path, char: str) -> None:
    """R1-1: the snapshot record the event stream emits, measured through ``_sse_data``.

    On 2a397de74 these rows fit the 4 MiB page bound in UTF-8 bytes, and the
    SSE record (``json.dumps`` escapes non-ASCII: 6 B per 2-byte character,
    12 B per emoji) was ~3x that — over the UI's 8 MiB per-record cap. CJK
    (3 B -> 6 B escaped, 2x) is covered too.
    """
    from local_operator.server.routes.desktop_sessions import _sse_data

    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    directory = tmp_path / "sessions" / sid
    directory.mkdir(parents=True, exist_ok=True)
    # ~400 KB of UTF-8 per row whatever the width (200k Cyrillic, ~133k CJK,
    # 100k emoji), so twelve rows overrun the page bound and force has_more.
    count = 400_000 // len(char.encode("utf-8"))
    rows = [*_opening(1), *(_text_row(f"t{n}", 200.0 + n, char, count) for n in range(12))]
    (directory / TRANSCRIPT_FILENAME).write_text(
        "\n".join(row.to_json() for row in rows) + "\n", encoding="utf-8"
    )
    async with pool.session(sid) as bridge:
        sub = bridge.subscribe()
        stream = bridge.events(sub, epoch=bridge.epoch, after_seq=bridge.sequence)
        assert (await anext(stream))["type"] == "open"
        snapshot = await anext(stream)
        assert snapshot["type"] == "snapshot"
        record = _sse_data(snapshot).encode("utf-8")
        assert len(record) <= UI_RECORD_CAP, len(record)
        history = snapshot["payload"]["history"]
        assert history["entries"][-1]["id"] == "t11"
        assert history["has_more"] is True
        await stream.aclose()


@pytest.mark.asyncio
async def test_a_bare_anchored_page_keeps_its_anchor(tmp_path: Path) -> None:
    """R1-5: ``around_id`` with ``after>0`` spent the budget on newer rows and lost the anchor."""
    async with _Harness(tmp_path) as harness:
        rows = [_sized_tool_row(f"t{n}", 200.0 + n, 1_400_000) for n in range(8)]
        harness.seed([*_opening(1), *rows])
        page = (
            await harness.client.get(
                f"{harness.base}/history", params={"around_id": "t1", "before": 1, "after": 6}
            )
        ).json()["result"]
        ids = [row["id"] for row in page["entries"]]
        assert "t1" in ids, ids
        assert ids == sorted(ids, key=lambda i: int(i[1:])), "contiguous, oldest first"
        assert page["has_newer"] is True, "newer rows were left out"
        assert sum(len(json.dumps(row)) for row in page["entries"]) <= 4 * 1024 * 1024


@pytest.mark.asyncio
async def test_the_documented_stale_ref_recovery_read_mints_a_working_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1-3: after a serve restart (new ref secret) the old ref 404s; the documented
    ``around_id=<row id>&before=0&after=0`` read mints a fresh ref that resolves."""
    from local_operator.session import row_clip

    async with _Harness(tmp_path) as harness:
        harness.seed([*_opening(), _sized_tool_row("huge", 200.0, C4_ROW_BYTES)])
        page = (await harness.client.get(f"{harness.base}/history")).json()["result"]
        stale = _newest(page)["truncated"]
        monkeypatch.setattr(row_clip, "_REF_KEY", b"\x01" * 32)  # a restarted process
        gone = await harness.client.get(f"{harness.base}/history/rows/{stale['ref']}")
        assert gone.status_code == 404
        again = (
            await harness.client.get(
                f"{harness.base}/history", params={"around_id": "huge", "before": 0, "after": 0}
            )
        ).json()["result"]
        [row] = again["entries"]
        fresh = row["truncated"]
        assert row["id"] == "huge" and fresh["ref"] != stale["ref"]
        assert fresh["bytes_total"] == stale["bytes_total"]
        data, _ = await _fetch_all(harness, fresh["ref"], step=1_048_576)
        assert len(data) == fresh["bytes_total"]
