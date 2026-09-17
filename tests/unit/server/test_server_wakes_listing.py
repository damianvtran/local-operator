"""``GET /v1/desktop/wakes``: the machine-wide listing.

Every case here is driven over HTTP against the real route, because the things
that can go wrong are all at the boundary: the index is a set of files other
processes write (so absence, corruption, an unknown schema and a torn name all
have to degrade to a ROW rather than to a 500), and the flags the page renders
have to be the supervisor's own predicates rather than a second derivation of
them.

The distinction this file exists to hold is between the two EMPTY answers: a
store with no wakes is a statement about the user's machine, and a store this
process could not read is a statement about the read. Only one of them is safe
to render as "no scheduled tasks".
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.server.routes import desktop_wakes

TOKEN = "desktop-wakes-listing-token"
FUTURE = int(time.time() * 1000) + 3_600_000
PAST = int(time.time() * 1000) - 1_800_000
#: Past ``wakes.supervisor.STALE_AFTER_S`` (7 days), where the supervisor stops
#: engaging and leaves the wake to the session's own catch-up.
ANCIENT = int(time.time() * 1000) - 8 * 24 * 3600_000


@pytest_asyncio.fixture
async def desktop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    app = FastAPI()
    app.include_router(desktop_wakes.router)
    app.state.config_manager = ConfigManager(tmp_path)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client, tmp_path


def _session(root: Path, session_id: str, *, transcript: bool = True) -> Path:
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    if transcript:
        (directory / "transcript.jsonl").write_text(
            json.dumps(
                {
                    "id": "m1",
                    "ts": 1.0,
                    "type": "message",
                    "payload": {
                        "kind": "message",
                        "role": "user",
                        "content": [{"type": "text", "text": "the invoices workspace"}],
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
    return directory


def _entry(
    root: Path,
    session_id: str,
    *,
    body_session_id: str | None = None,
    cwd: str = "/work/here",
    rows: list[dict[str, Any]] | None = None,
    **extra: object,
) -> Path:
    entry = {
        "schema": 1,
        "session_id": body_session_id or session_id,
        "cwd": cwd,
        "updated_at": 1_700_000_000_000,
        "schedules": rows if rows is not None else [_row("w1")],
    }
    entry.update(extra)
    path = root / "wakes" / f"{session_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry), encoding="utf-8")
    return path


def _row(wake_id: str = "w1", *, due: int | None = None, **extra: object) -> dict[str, Any]:
    row = {
        "id": wake_id,
        "message": f"message {wake_id}",
        "next_due_at": FUTURE if due is None else due,
        "every_ms": 3_600_000,
        "until_at": None,
        "limit": None,
        "fired_count": 0,
        "created_at": 1,
    }
    row.update(extra)
    return row


async def _list(client: AsyncClient, **params: str | int) -> dict[str, Any]:
    response = await client.get("/v1/desktop/wakes", params=params or None)
    assert response.status_code == 200, response.text
    return response.json()["result"]


@pytest.mark.asyncio
async def test_an_absent_store_is_an_empty_list_and_not_an_error(desktop) -> None:
    """No ``wakes/`` directory is the ordinary state of a machine that has
    never scheduled anything, and it must not read as a failed read."""
    client, _root = desktop

    listing = await _list(client)

    assert listing["entries"] == []
    assert listing["total"] == 0
    assert listing["truncated"] is False
    assert listing["read_error"] is False
    assert "supervisor" in listing


@pytest.mark.asyncio
async def test_an_unreadable_store_reports_the_read_rather_than_no_schedules(desktop) -> None:
    """The one case where "no scheduled tasks" would be a claim this process has
    not earned: the directory is there and cannot be listed."""
    client, root = desktop
    _entry(root, "aaaaaaaaaaaa")
    wakes = root / "wakes"
    wakes.chmod(0o000)
    try:
        listing = await _list(client)
    finally:
        wakes.chmod(0o700)

    assert listing["entries"] == []
    assert listing["total"] == 0
    assert listing["read_error"] is True


@pytest.mark.asyncio
async def test_a_corrupt_entry_costs_one_row_and_an_unknown_schema_is_skipped(desktop) -> None:
    """``read_entry``/``read_index`` already treat both as absent — the owning
    session rewrites the file on its next open — so the listing inherits that
    instead of inventing a repair of its own."""
    client, root = desktop
    _session(root, "aaaaaaaabbbb")
    _entry(root, "aaaaaaaabbbb")
    (root / "wakes" / "ccccccccdddd.json").write_text("{not json", encoding="utf-8")
    _entry(root, "eeeeeeeeffff", rows=[_row("w1")])
    (root / "wakes" / "eeeeeeeeffff.json").write_text(
        json.dumps({"schema": 99, "session_id": "eeeeeeeeffff", "schedules": []}),
        encoding="utf-8",
    )

    listing = await _list(client)

    assert [entry["session_id"] for entry in listing["entries"]] == ["aaaaaaaabbbb"]
    assert listing["total"] == 1


@pytest.mark.asyncio
async def test_the_filename_wins_over_a_disagreeing_session_id_in_the_body(desktop) -> None:
    """A copied or hand-edited file. The filename is the lookup key and the key
    the owning session will overwrite, so the row is not duplicated."""
    client, root = desktop
    _session(root, "aaaaaaaaaaaa")
    _entry(root, "aaaaaaaaaaaa", body_session_id="bbbbbbbbbbbb")

    listing = await _list(client)

    assert [entry["session_id"] for entry in listing["entries"]] == ["aaaaaaaaaaaa"]
    assert listing["total"] == 1


@pytest.mark.asyncio
async def test_a_ghost_is_flagged_and_still_listed(desktop) -> None:
    """The index outlives the session directory. The supervisor refuses to
    engage a ghost, so a row that claimed "armed" would contradict the process
    that fires it."""
    client, root = desktop
    _entry(root, "aaaaaaaaaaaa")  # no session directory at all

    listing = await _list(client)

    entry = listing["entries"][0]
    assert entry["ghost"] is True
    assert entry["dormant"] is False


@pytest.mark.asyncio
async def test_a_dormant_session_is_flagged_and_can_be_excluded(desktop) -> None:
    """A stopped session's wakes stay armed and do not fire until it is
    reopened, which is a different fact from "gone" — so it is a different
    flag, and the page can ask for the live ones alone."""
    client, root = desktop
    _session(root, "aaaaaaaaaaaa")
    _entry(root, "aaaaaaaaaaaa", stopped_at=1234)

    with_dormant = await _list(client)
    without = await _list(client, include_dormant="false")

    assert with_dormant["entries"][0]["dormant"] is True
    assert with_dormant["entries"][0]["ghost"] is False
    assert without["entries"] == []


@pytest.mark.asyncio
async def test_overdue_and_stale_are_both_reported_with_the_supervisor_predicate(
    desktop,
) -> None:
    """A row can be late without being given up on. The stale verdict has to be
    the supervisor's own 7-day bound, imported rather than re-derived: the CLI
    once said "1 armed, 10m overdue" about a wake the supervisor had retired
    over."""
    client, root = desktop
    _session(root, "aaaaaaaaaaaa")
    _entry(root, "aaaaaaaaaaaa", rows=[_row("w1", due=PAST), _row("w2", due=ANCIENT)])

    listing = await _list(client)

    rows = {row["id"]: row for row in listing["entries"][0]["schedules"]}
    assert rows["w1"]["overdue_s"] > 1700
    assert rows["w1"]["stale"] is False
    assert rows["w2"]["stale"] is True
    # The entry's own "next due" is the soonest row across the entry, which is
    # the STALE one here: the flag is per schedule on purpose, and an entry
    # holding a given-up-on one-shot beside a live watch must not let the
    # stale sibling speak for the rest.
    assert listing["entries"][0]["next_due_at"] == ANCIENT


@pytest.mark.asyncio
async def test_a_failed_name_read_falls_back_to_the_id_and_never_fails_the_listing(
    desktop, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A name is decoration. The floor is not: a nameless row is one the user
    cannot identify, and it must not cost the other sessions their rows."""
    import local_operator.resume as resume_module

    client, root = desktop
    _session(root, "aaaaaaaaaaaa")
    _entry(root, "aaaaaaaaaaaa", cwd="/work/invoices")

    def boom(directory: Path, **kwargs: object) -> str:
        raise OSError("the transcript could not be read")

    monkeypatch.setattr(resume_module, "session_name", boom)

    listing = await _list(client)

    assert listing["entries"][0]["name"] == "aaaaaaaa (invoices)"


@pytest.mark.asyncio
async def test_a_name_is_read_with_the_shared_resolver(desktop) -> None:
    """Same function the picker and the catalog use, so one conversation cannot
    be named two ways on two surfaces."""
    client, root = desktop
    _session(root, "aaaaaaaaaaaa")
    _entry(root, "aaaaaaaaaaaa")

    listing = await _list(client)

    assert listing["entries"][0]["name"] == "the invoices workspace"


@pytest.mark.asyncio
async def test_rows_are_sorted_soonest_first_and_paged_with_a_visible_truth(desktop) -> None:
    """Soonest first, undateable last, ties by id — the same ordering rule the
    run pane's wake list applies, so the two surfaces agree when both are on
    screen. ``total`` counts what was NOT sent."""
    client, root = desktop
    _session(root, "aaaaaaaaaaaa")
    _session(root, "bbbbbbbbbbbb")
    _session(root, "cccccccccccc")
    _entry(root, "aaaaaaaaaaaa", rows=[_row("w1", due=FUTURE + 60_000)])
    _entry(root, "bbbbbbbbbbbb", rows=[_row("w1", due=FUTURE)])
    _entry(root, "cccccccccccc", rows=[])

    listing = await _list(client, limit=2)

    assert [entry["session_id"] for entry in listing["entries"]] == [
        "bbbbbbbbbbbb",
        "aaaaaaaaaaaa",
    ]
    assert listing["total"] == 3
    assert listing["truncated"] is True
    # An entry whose rows are all unreadable sorts last rather than vanishing.
    full = await _list(client)
    assert [entry["session_id"] for entry in full["entries"]][-1] == "cccccccccccc"


@pytest.mark.asyncio
async def test_the_supervisor_block_answers_whether_anything_is_watching(desktop) -> None:
    """The index alone cannot answer "will this fire": the supervisor is a
    separate process whose states are invisible from the files."""
    client, _root = desktop

    listing = await _list(client)

    supervisor = listing["supervisor"]
    assert set(supervisor) >= {"supported", "running", "detail"}


@pytest.mark.asyncio
async def test_an_unknown_query_parameter_is_refused(desktop) -> None:
    """The listing's own admission: an out-of-range page is a client bug, not a
    silently clamped answer."""
    client, _root = desktop

    assert (await client.get("/v1/desktop/wakes", params={"limit": 0})).status_code == 422
    assert (await client.get("/v1/desktop/wakes", params={"limit": 501})).status_code == 422


@pytest.mark.asyncio
async def test_the_route_requires_the_desktop_bearer(desktop) -> None:
    """Same door as every sibling desktop module — a session-scoped surface
    that mutates automation must not be reachable without it."""
    _client, root = desktop
    app = FastAPI()
    app.include_router(desktop_wakes.router)
    app.state.config_manager = ConfigManager(root)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as bare:
        assert (await bare.get("/v1/desktop/wakes")).status_code in (401, 403)
