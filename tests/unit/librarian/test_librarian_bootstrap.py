"""``librarian.bootstrap.ensure_session`` — creation, identity, zero footprint.

The visibility promise is pinned here by SNAPSHOT and by the ABSENCE of
``origin.json``: the Librarian must be an ordinary user session on every
surface (design §5), which is exactly what a bespoke origin would prevent.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from local_operator.librarian import bootstrap, state
from tests.unit.aida.conftest import tree, write_config


@pytest.mark.asyncio
async def test_ensure_creates_her_session_once_and_returns_the_same_id(
    isolated_root: Path,
) -> None:
    first = await bootstrap.ensure_session(isolated_root)
    second = await bootstrap.ensure_session(isolated_root)

    assert first and second == first
    directory = isolated_root / "sessions" / first
    assert directory.is_dir()
    transcript = (directory / "transcript.jsonl").read_text(encoding="utf-8")
    # The title rides both channels, and the birth entry is the activity
    # clock's reason to rank the row and make the directory resumable.
    assert '"conversation_name"' in transcript and '"librarian_session"' in transcript
    assert json.loads((directory / "attachment.json").read_text())["agent"] == "librarian"
    title = json.loads((directory / "title.json").read_text())
    assert title["text"] == "Librarian" and title["user_set"] is True
    assert (directory / "created_at.json").exists()
    # NO ORIGIN: a bespoke origin is hidden by resume.is_user_session_origin,
    # which is what would make her invisible on the phone.
    assert not (directory / "origin.json").exists()
    # State names her, the pin store carries her, and the cadence is armed.
    assert state.session_id_of(isolated_root) == first
    assert first in json.loads((isolated_root / "sidebar-pins.json").read_text())
    entry = json.loads((isolated_root / "wakes" / f"{first}.json").read_text())
    assert [row["id"] for row in entry["schedules"]] == ["librarian-cadence"]


@pytest.mark.asyncio
async def test_the_configured_name_is_what_creation_writes(isolated_root: Path) -> None:
    write_config(isolated_root, {"librarian": {"name": "Archivist"}})
    sid = await bootstrap.ensure_session(isolated_root)
    directory = isolated_root / "sessions" / str(sid)
    assert json.loads((directory / "title.json").read_text())["text"] == "Archivist"
    assert bootstrap.display_name(isolated_root) == "Archivist"


@pytest.mark.asyncio
async def test_a_renamed_existing_session_is_reconciled_on_the_next_ensure(
    isolated_root: Path,
) -> None:
    sid = await bootstrap.ensure_session(isolated_root)
    write_config(isolated_root, {"librarian": {"name": "Archivist"}})
    assert await bootstrap.ensure_session(isolated_root) == sid
    title = json.loads((isolated_root / "sessions" / str(sid) / "title.json").read_text())
    assert title["text"] == "Archivist"


@pytest.mark.asyncio
async def test_disabled_by_env_writes_nothing(isolated_root: Path, monkeypatch) -> None:
    monkeypatch.setenv(state.ENV_DISABLE, "1")
    before = tree(isolated_root)
    assert await bootstrap.ensure_session(isolated_root) is None
    assert tree(isolated_root) == before


@pytest.mark.asyncio
async def test_disabled_by_config_writes_nothing(isolated_root: Path) -> None:
    write_config(isolated_root, {"librarian": {"enabled": False}})
    before = tree(isolated_root)
    assert await bootstrap.ensure_session(isolated_root) is None
    assert tree(isolated_root) == before


@pytest.mark.asyncio
async def test_a_recreated_session_reaps_the_previous_incarnations_rows(
    isolated_root: Path,
) -> None:
    from local_operator.harness.wake_types import WakeSchedule
    from local_operator.wakes import store as wake_store

    # A previous incarnation: a row on the index whose session directory is
    # gone. Nothing can ever fire it, and it is exactly the litter the boot
    # reaps.
    wake_store.write_entry(
        isolated_root,
        "deadbeefcafe",
        cwd=str(isolated_root),
        schedules=[
            WakeSchedule(id="librarian-cadence", message="old", next_due_at=1, created_at=1)
        ],
    )

    sid = await bootstrap.ensure_session(isolated_root)
    assert sid
    assert not (isolated_root / "wakes" / "deadbeefcafe.json").exists()


@pytest.mark.asyncio
async def test_two_concurrent_ensures_mint_one_session(isolated_root: Path) -> None:
    first, second = await asyncio.gather(
        bootstrap.ensure_session(isolated_root), bootstrap.ensure_session(isolated_root)
    )
    # One wins outright; the other either observes the winner or reports a
    # lock refusal (best-effort by contract), so at most one session exists.
    sessions = [p for p in (isolated_root / "sessions").iterdir() if p.is_dir()]
    assert len(sessions) == 1
    assert {first, second} & {sessions[0].name}
