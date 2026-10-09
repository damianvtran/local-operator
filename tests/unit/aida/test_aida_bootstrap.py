"""``aida.bootstrap.ensure_session`` — creation, idempotence, zero footprint.

The R17/R18 promise is pinned here by SNAPSHOT, not by a return value: a
disabled install must gain no file of any kind, and the test compares the whole
root tree before and after rather than trusting ``ensure_session`` to say so.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from local_operator import aida
from local_operator.aida import state
from tests.unit.aida.conftest import mark_met, tree, write_config


@pytest.mark.asyncio
async def test_ensure_creates_her_session_once_and_returns_the_same_id(isolated_root: Path) -> None:
    # An install she has already met: the cadence half below is armed by the
    # ensure only once the greeting is delivered (onboarding.cadence_allowed).
    mark_met(isolated_root)
    first = await aida.ensure_session(isolated_root)
    second = await aida.ensure_session(isolated_root)

    assert first and second == first
    directory = isolated_root / "sessions" / first
    assert directory.is_dir()
    # The directory is REAL: a transcript with her title and birth entry, the
    # role attachment the load path restores, and the created-at sidecar.
    transcript = (directory / "transcript.jsonl").read_text(encoding="utf-8")
    assert '"conversation_name"' in transcript and '"aida_session"' in transcript
    assert json.loads((directory / "attachment.json").read_text())["agent"] == "aida"
    title = json.loads((directory / "title.json").read_text())
    assert title["text"] == "Aida" and title["user_set"] is True
    # State names her, the pin store carries her, and the cadence is armed.
    assert state.session_id_of(isolated_root) == first
    assert first in json.loads((isolated_root / "sidebar-pins.json").read_text())
    entry = json.loads((isolated_root / "wakes" / f"{first}.json").read_text())
    ids = [row["id"] for row in entry["schedules"]]
    assert ids == ["aida-cadence"]


@pytest.mark.asyncio
async def test_her_id_is_recorded_before_her_directory_exists(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The create ORDER is load-bearing: a scan mid-create must not see her as a stranger.

    Two boot hooks run concurrently — this ensure, and the TUI's first-run route,
    which asks ``onboarding.cadence_allowed`` whether the install already has
    conversations and SETTLES the ledger on that answer. ``other_user_sessions``
    excludes the one session ``aida/state.json`` names, so a create that
    materialises the directory first leaves a window in which a scan excludes
    NOTHING and counts her own born journal (title + birth rows — never a
    message) as the operator's. On CI that window was lost every run: a fresh
    isolated root came out of its first boot with the greeting ``skipped``,
    stamped in the same millisecond as ``state.created_at`` (``tui-e2e``,
    ubuntu-latest, 2026-10-09).
    """
    from local_operator.aida import bootstrap

    observed: list[str | None] = []
    real_create = bootstrap._create_session_dir

    async def spy(config_dir: Path, session_id: str) -> None:
        # Exactly what a concurrent scan reads, at the instant before the
        # directory that scan would count comes into existence.
        observed.append(state.session_id_of(config_dir))
        await real_create(config_dir, session_id)

    monkeypatch.setattr(bootstrap, "_create_session_dir", spy)
    session_id = await aida.ensure_session(isolated_root)

    assert session_id is not None
    assert observed == [session_id], (
        "her id must already be recorded when her directory is materialised, "
        "or a concurrent sessions scan counts her as the operator's conversation"
    )


@pytest.mark.asyncio
async def test_concurrent_first_invocations_mint_one_session(isolated_root: Path) -> None:
    """Two callers racing the create converge on ONE conversation.

    The ensure lock is what makes this true; without it both would find no
    state, mint an id, and one would clobber the other's state file while both
    directories survived.
    """
    results = await asyncio.gather(
        aida.ensure_session(isolated_root),
        aida.ensure_session(isolated_root),
    )
    alive = [session_id for session_id in results if session_id]
    assert alive, "at least one caller must get an id"
    assert len(set(alive)) == 1
    assert len(list((isolated_root / "sessions").iterdir())) == 1


@pytest.mark.asyncio
async def test_disabled_env_leaves_zero_footprint(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``LOCAL_OPERATOR_NO_AIDA=1`` writes nothing at all (R17)."""
    monkeypatch.setenv("LOCAL_OPERATOR_NO_AIDA", "1")
    before = tree(isolated_root)

    assert await aida.ensure_session(isolated_root) is None

    assert tree(isolated_root) == before
    assert not (isolated_root / "aida").exists()


@pytest.mark.asyncio
async def test_disabled_config_key_leaves_zero_footprint(isolated_root: Path) -> None:
    """``aida.enabled = false`` is the same promise via the config key (R18)."""
    write_config(isolated_root, {"aida": {"enabled": False}})
    before = tree(isolated_root)

    assert await aida.ensure_session(isolated_root) is None

    assert tree(isolated_root) == before


@pytest.mark.asyncio
async def test_env_switch_truthiness_matches_the_house_convention(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for falsy in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("LOCAL_OPERATOR_NO_AIDA", falsy)
        assert state.env_disabled() is False
    for truthy in ("1", "true", "yes", "on"):
        monkeypatch.setenv("LOCAL_OPERATOR_NO_AIDA", truthy)
        assert state.env_disabled() is True
