"""``GET /v1/desktop/monitors``: the machine-wide listing.

Every case here is driven over HTTP against the real route, because the things
that can go wrong are all at the boundary: the index is a set of files other
processes write (so absence, corruption, an unknown schema and a torn name all
have to degrade to a ROW rather than to a 500), and the derived fields the page
renders have to be the same derivations the CLI makes rather than a second
opinion about them.

The distinction this file exists to hold is between the two EMPTY answers: a
store with no monitors is a statement about the user's machine, and a store
this process could not read is a statement about the read. Only one of them is
safe to render as "no standing watches".
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
from local_operator.server.routes import desktop_monitors

TOKEN = "desktop-monitors-listing-token"
FUTURE = int(time.time() * 1000) + 3_600_000
PAST = int(time.time() * 1000) - 1_800_000


@pytest_asyncio.fixture
async def desktop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    app = FastAPI()
    app.include_router(desktop_monitors.router)
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
        "monitors": rows if rows is not None else [_row("m1")],
    }
    entry.update(extra)
    path = root / "monitors" / f"{session_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry), encoding="utf-8")
    return path


def _row(monitor_id: str = "m1", *, due: int | None = None, **extra: object) -> dict[str, Any]:
    row = {
        "id": monitor_id,
        "name": f"watch {monitor_id}",
        "tool": "bash",
        "arguments": {"command": "ls"},
        "every_ms": 60_000,
        "until_at": None,
        "description": "",
        "next_due_at": FUTURE if due is None else due,
        "last_check_at": 0,
        "checks": 0,
        "deliveries": 0,
        "consecutive_failures": 0,
        "disabled": False,
        "disabled_reason": "",
        "created_at": 1,
    }
    row.update(extra)
    return row


async def _list(client: AsyncClient, **params: str | int) -> dict[str, Any]:
    response = await client.get("/v1/desktop/monitors", params=params or None)
    assert response.status_code == 200, response.text
    return response.json()["result"]


@pytest.mark.asyncio
async def test_an_absent_store_is_an_empty_list_and_not_an_error(desktop) -> None:
    """No ``monitors/`` directory is the ordinary state of a machine that has
    never armed a watch, and it must not read as a failed read."""
    client, _root = desktop

    listing = await _list(client)

    assert listing["entries"] == []
    assert listing["total"] == 0
    assert listing["truncated"] is False
    assert listing["read_error"] is False
    # The wake surface's supervisor block has no monitor twin — a monitor never
    # engages a cold session (§10.4), so there is no watcher to report on.
    assert "supervisor" not in listing


@pytest.mark.asyncio
async def test_an_unreadable_store_reports_the_read_rather_than_no_watches(desktop) -> None:
    """The one case where "no standing watches" would be a claim this process
    has not earned: the directory is there and cannot be listed."""
    client, root = desktop
    _entry(root, "aaaaaaaaaaaa")
    monitors = root / "monitors"
    monitors.chmod(0o000)
    try:
        listing = await _list(client)
    finally:
        monitors.chmod(0o700)

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
    (root / "monitors" / "ccccccccdddd.json").write_text("{not json", encoding="utf-8")
    _entry(root, "eeeeeeeeffff")
    (root / "monitors" / "eeeeeeeeffff.json").write_text(
        json.dumps({"schema": 99, "session_id": "eeeeeeeeffff", "monitors": []}),
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
    """The index outlives the session directory. Nothing can tick a ghost, so a
    row that claimed "armed" would describe a watch that cannot run."""
    client, root = desktop
    _entry(root, "aaaaaaaaaaaa")  # no session directory at all

    listing = await _list(client)

    entry = listing["entries"][0]
    assert entry["ghost"] is True
    assert entry["dormant"] is False
    # And the row still carries its watch, so it can be cancelled from here.
    assert [row["id"] for row in entry["monitors"]] == ["m1"]


@pytest.mark.asyncio
async def test_a_dormant_session_is_flagged_and_can_be_excluded(desktop) -> None:
    """A stopped session's monitors stay armed and do not tick until it is
    reopened, which is a different fact from "gone" — so it is a different
    flag, and the page can ask for the live ones alone."""
    client, root = desktop
    _session(root, "aaaaaaaaaaaa")
    _entry(root, "aaaaaaaaaaaa", stopped_at=1234)

    with_dormant = await _list(client)
    without = await _list(client, include_dormant="false")

    assert with_dormant["entries"][0]["dormant"] is True
    assert with_dormant["entries"][0]["ghost"] is False
    # The park is entry-level, and it reaches the row's state word too — the
    # CLI's precedence (`cli._monitor_state_word`), so the two surfaces cannot
    # describe one watch two ways.
    assert with_dormant["entries"][0]["monitors"][0]["state"] == "dormant"
    assert without["entries"] == []


@pytest.mark.asyncio
async def test_the_state_word_follows_the_cli_precedence(desktop) -> None:
    """Dormancy beats disabled (nothing is SUPPOSED to run, so a failure word
    would point at the wrong remedy); disabled beats the clock (a watch that
    does not tick must not read as merely late); expiry is the next fall."""
    client, root = desktop
    _session(root, "aaaaaaaaaaaa")
    _entry(
        root,
        "aaaaaaaaaaaa",
        rows=[
            _row("m1"),
            _row("m2", disabled=True, disabled_reason="5 failed"),
            _row("m3", until_at=PAST),
            _row("m4", due=PAST),
        ],
    )

    listing = await _list(client)

    states = {row["id"]: row["state"] for row in listing["entries"][0]["monitors"]}
    assert states == {"m1": "armed", "m2": "disabled", "m3": "expired", "m4": "armed"}
    # A late watch is still armed: the due time moves only on change events
    # (§10.2), so lateness between events is ordinary, not a state.
    rows = {row["id"]: row for row in listing["entries"][0]["monitors"]}
    assert rows["m4"]["due_in_s"] < 0
    assert rows["m1"]["due_in_s"] > 0
    assert rows["m2"]["disabled_reason"] == "5 failed"
    assert rows["m4"]["next_due_at"] == PAST


@pytest.mark.asyncio
async def test_the_clock_fields_degrade_to_null_rather_than_zero(desktop) -> None:
    """A defaulted stamp is a measurement the server never made: a watch with
    no recorded due time says so, and one that has never checked says so too —
    both are states a renderer shows differently from "due at epoch 0"."""
    client, root = desktop
    _session(root, "aaaaaaaaaaaa")
    _entry(root, "aaaaaaaaaaaa", rows=[_row("m1", due=None, next_due_at=None)])

    listing = await _list(client)

    row = listing["entries"][0]["monitors"][0]
    assert row["next_due_at"] is None
    assert row["due_in_s"] is None
    assert row["last_check_age_s"] is None
    assert listing["entries"][0]["next_due_at"] is None


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
async def test_entries_and_rows_are_sorted_soonest_first_and_paged_visibly(desktop) -> None:
    """Soonest first, undateable last, ties by id — the ordering rule both
    stores' scans use, so the two surfaces agree when both are on screen.
    ``total`` counts what was NOT sent, and a row list orders the same way
    inside its entry."""
    client, root = desktop
    _session(root, "aaaaaaaaaaaa")
    _session(root, "bbbbbbbbbbbb")
    _session(root, "cccccccccccc")
    _entry(
        root,
        "aaaaaaaaaaaa",
        rows=[_row("m2", due=FUTURE + 60_000), _row("m1", due=FUTURE + 120_000)],
    )
    _entry(root, "bbbbbbbbbbbb", rows=[_row("m1", due=FUTURE)])
    _entry(root, "cccccccccccc", rows=[])

    listing = await _list(client, limit=2)

    assert [entry["session_id"] for entry in listing["entries"]] == [
        "bbbbbbbbbbbb",
        "aaaaaaaaaaaa",
    ]
    assert [row["id"] for row in listing["entries"][1]["monitors"]] == ["m2", "m1"]
    assert listing["total"] == 3
    assert listing["truncated"] is True
    # An entry whose rows are all unreadable sorts last rather than vanishing.
    full = await _list(client)
    assert [entry["session_id"] for entry in full["entries"]][-1] == "cccccccccccc"


@pytest.mark.asyncio
async def test_an_out_of_range_page_is_refused(desktop) -> None:
    """The listing's own admission: an out-of-range page is a client bug, not a
    silently clamped answer."""
    client, _root = desktop

    assert (await client.get("/v1/desktop/monitors", params={"limit": 0})).status_code == 422
    assert (await client.get("/v1/desktop/monitors", params={"limit": 501})).status_code == 422


@pytest.mark.asyncio
async def test_the_route_requires_the_desktop_bearer(desktop) -> None:
    """Same door as every sibling desktop module — a session-scoped surface
    that mutates automation must not be reachable without it."""
    _client, root = desktop
    app = FastAPI()
    app.include_router(desktop_monitors.router)
    app.state.config_manager = ConfigManager(root)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as bare:
        assert (await bare.get("/v1/desktop/monitors")).status_code in (401, 403)


@pytest.mark.asyncio
async def test_a_row_carries_the_health_hint_and_the_unavailable_since(desktop) -> None:
    """§D6: the desktop page renders the shared hint rather than deriving its
    own, so one monitor cannot read as healthy here and idle on the CLI.
    """
    client, root = desktop
    _session(root, "mhchk1")
    _entry(
        root,
        "mhchk1",
        rows=[
            _row(
                "m1",
                checks=9,
                deliveries=0,
                created_at=int(time.time() * 1000) - 7_200_000,
            ),
            _row(
                "m2",
                due=FUTURE,
                # An unavailable episode, six minutes in.
                unavailable_since=int(time.time() * 1000) - 360_000,
                checks=4,
                deliveries=1,
                created_at=int(time.time() * 1000) - 7_200_000,
            ),
        ],
    )

    response = await client.get("/v1/desktop/monitors")
    assert response.status_code == 200, response.text
    payload = response.json()["result"]
    rows = {row["id"]: row for entry in payload["entries"] for row in entry["monitors"]}
    assert rows["m1"]["health"] == (
        "9 checks, 0 deliveries — nothing has changed (confirm the call observes what you expect)"
    )
    assert rows["m2"]["health"] is not None and rows["m2"]["health"].startswith(
        "tool unavailable since "
    )
    assert rows["m2"]["unavailable_since"] > 0
    assert rows["m1"]["unavailable_since"] == 0


@pytest.mark.asyncio
async def test_a_healthy_row_carries_no_hint(desktop) -> None:
    client, root = desktop
    _session(root, "mhchk2")
    _entry(
        root,
        "mhchk2",
        rows=[_row("m1", checks=6, deliveries=2, created_at=int(time.time() * 1000) - 7_200_000)],
    )

    response = await client.get("/v1/desktop/monitors")
    assert response.status_code == 200
    row = response.json()["result"]["entries"][0]["monitors"][0]
    assert row["health"] is None
