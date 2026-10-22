"""``librarian.bootstrap.ensure_session`` — creation, identity, zero footprint.

The visibility promise is pinned here by SNAPSHOT and by the ABSENCE of
``origin.json``: the Librarian must be an ordinary user session on every
surface (design §5), which is exactly what a bespoke origin would prevent.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
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


@pytest.mark.asyncio
async def test_a_concurrent_first_boot_keeps_both_assistants_pinned(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The measured defect: both bootstraps pin, one pin survives.

    The barrier releases both threads together and the injected delay sits
    between the pin index's read and its replace — the exact window the lock
    closes. On the unlocked store this test fails deterministically (both read
    ``[]``, the later replace wins); with the lock the peer waits and then
    appends to what the first wrote.
    """
    monkeypatch.delenv("LOCAL_OPERATOR_NO_AIDA", raising=False)
    import local_operator.tui.sidebar_pins as pins
    from local_operator import aida

    barrier = threading.Barrier(2)
    original_set = pins.set_pin
    original_read = pins.read_pins

    def synchronized_set(
        config_dir: Path, session_id: str, pinned: bool, *, source: str = pins.SOURCE_USER
    ) -> bool:
        barrier.wait(timeout=10)
        return original_set(config_dir, session_id, pinned, source=source)

    def slow_read(config_dir: Path) -> list[str]:
        time.sleep(0.05)
        return original_read(config_dir)

    monkeypatch.setattr(pins, "set_pin", synchronized_set)
    monkeypatch.setattr(pins, "read_pins", slow_read)

    aida_id, librarian_id = await asyncio.gather(
        aida.ensure_session(isolated_root), bootstrap.ensure_session(isolated_root)
    )
    assert aida_id and librarian_id and aida_id != librarian_id
    assert set(pins.read_pins(isolated_root)) == {aida_id, librarian_id}


@pytest.mark.asyncio
async def test_a_later_boot_restores_a_missing_pin(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second half of the reported bug: a lost pin was never re-asserted.

    The ensure used to pin only in its create branch, so a pin file that lost
    her id while her session survived stayed wrong forever. Reproducing the
    end state (Aida pinned, Librarian unpinned) and booting again must put her
    back without minting a new session.
    """
    monkeypatch.delenv("LOCAL_OPERATOR_NO_AIDA", raising=False)
    import local_operator.tui.sidebar_pins as pins
    from local_operator import aida

    aida_id = await aida.ensure_session(isolated_root)
    librarian_id = await bootstrap.ensure_session(isolated_root)
    # Simulate the RACE-LOST state, not a human unpin: the low-level writer
    # leaves no deliberate-unpin marker, which is exactly the shape the bug
    # produced.
    assert aida_id is not None
    pins._write_pins(isolated_root, [aida_id])
    assert pins.read_pins(isolated_root) == [aida_id]

    assert await bootstrap.ensure_session(isolated_root) == librarian_id
    assert set(pins.read_pins(isolated_root)) == {aida_id, librarian_id}


@pytest.mark.asyncio
async def test_an_explicit_unpin_survives_the_next_boot(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reassert must repair a lost pin, never overrule the person.

    A desktop/TUI unpin is a human write and records intent; the next boot must
    leave the row off (QA Q1's "respecting deliberate user unpins").
    """
    monkeypatch.delenv("LOCAL_OPERATOR_NO_AIDA", raising=False)
    import local_operator.tui.sidebar_pins as pins

    librarian_id = await bootstrap.ensure_session(isolated_root)
    assert librarian_id is not None
    assert pins.set_pin(isolated_root, librarian_id, False) is False
    assert librarian_id in pins.explicitly_unpinned(isolated_root)

    assert await bootstrap.ensure_session(isolated_root) == librarian_id
    assert librarian_id not in pins.read_pins(isolated_root)
    # A human re-pin clears the intent and sticks.
    assert pins.set_pin(isolated_root, librarian_id, True) is True
    assert librarian_id not in pins.explicitly_unpinned(isolated_root)


def test_a_failed_create_records_a_failed_create_receipt(tmp_path: Path) -> None:
    from local_operator.librarian import bootstrap

    session_id = "lib111122222"
    directory = tmp_path / "sessions" / session_id
    directory.mkdir(parents=True)
    (directory / "created_at.json").write_text("{}", encoding="utf-8")

    bootstrap._discard_failed_create(tmp_path, session_id)

    assert not directory.exists()
    rows = [
        json.loads(line)
        for line in (tmp_path / "sessions" / ".cleanup-log.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["policy"] == "failed-create"
    assert rows[0]["session"] == session_id
    assert rows[0]["actor"] == "librarian"
