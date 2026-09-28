"""``aida.name`` — the display name, its validation, and the sync both ways.

The 2026-09-28 report this file pins: renaming her pinned conversation changed
the conversation and nothing else. The contract now is ONE config key that
every surface reads, kept in step with her conversation's title by the two
write paths exercised here (the session rename sync lives beside the other
session-side hooks, in ``test_aida_session_hooks.py``; the TUI's rename verb
in ``test_aida_tui.py``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator import settings_io
from local_operator.aida import naming
from local_operator.config import ConfigManager
from local_operator.resume import (
    read_title_state,
    stored_session_title,
    write_session_title,
)
from local_operator.session import naming as session_naming
from local_operator.session.transcript import read_latest_custom
from tests.unit.aida.conftest import write_config

HER_ID = "aaaa11112222"


def _her_session(root: Path, *, title: str = "Aida") -> Path:
    """A minimal version of her session directory: the two title channels."""
    directory = root / "sessions" / HER_ID
    directory.mkdir(parents=True, exist_ok=True)
    write_session_title(directory, title, user_set=True, past_names=[])
    return directory


def test_the_default_is_the_packaged_name(isolated_root: Path) -> None:
    assert naming.display_name(isolated_root) == "Aida"
    assert naming.display_name() != ""  # the no-argument form never raises either
    assert naming.DEFAULT_NAME == "Aida"


def test_the_caps_cannot_drift() -> None:
    """The registry keeps a literal; the real constants live in the consumers.

    ``settings_io`` deliberately stays off the aida package's import path, so
    the cap is a second literal there — pinned here the way
    ``_consumer_defaults`` pins the defaults. The session-title cap must agree
    too: a session rename syncs its title into this key, so a title the
    session layer accepted must never be refused by the config layer mid-sync.
    """
    assert settings_io._AIDA_NAME_MAX_CHARS == naming.MAX_NAME_CHARS
    assert naming.MAX_NAME_CHARS == session_naming.MAX_TITLE_CHARS


@pytest.mark.parametrize("value", ["", "   ", "\t\n", "x" * 81, "Bo\x1bb", "bell\x07"])
def test_invalid_names_are_refused_with_the_registry_refusal(
    value: str, isolated_root: Path
) -> None:
    """One sentence for one value, on the page and in the receipt alike."""
    setting = settings_io.BY_KEY["aida.name"]
    problem = settings_io.validate(setting, value)
    assert problem, value
    with pytest.raises(ValueError) as excinfo:
        naming.validate_name(value)
    assert str(excinfo.value) == problem
    # The write path refuses too, so no surface can store what validate
    # rejects — and the refusal cannot leave a config behind.
    with pytest.raises(ValueError):
        naming.set_name(isolated_root, value)
    assert naming.display_name(isolated_root) == "Aida"


def test_a_valid_name_normalizes_whitespace() -> None:
    assert naming.validate_name("  Maya   Chen  ") == "Maya Chen"
    assert settings_io.validate(settings_io.BY_KEY["aida.name"], "  Maya  ") is None


def test_set_name_writes_through_the_registry(isolated_root: Path) -> None:
    stored = naming.set_name(isolated_root, "  Maya ")
    assert stored == "Maya"
    # The consumer accessor: the nested path the readers use, never the flat one.
    assert ConfigManager(config_dir=isolated_root).get_nested_value(("aida", "name")) == "Maya"
    assert naming.display_name(isolated_root) == "Maya"


def test_sync_from_title_adopts_a_user_title(isolated_root: Path) -> None:
    assert naming.sync_config_name_from_title(isolated_root, "Sovereign") == "Sovereign"
    assert naming.display_name(isolated_root) == "Sovereign"


def test_sync_from_title_skips_an_unusable_title(isolated_root: Path) -> None:
    assert naming.sync_config_name_from_title(isolated_root, "   ") is None
    assert naming.sync_config_name_from_title(isolated_root, "x" * 81) is None
    assert naming.display_name(isolated_root) == "Aida"


def test_display_name_falls_back_on_a_stored_value_that_cannot_be_a_name(
    isolated_root: Path,
) -> None:
    """A hand-edited file reaches no writer's validation; the READER clamps."""
    write_config(isolated_root, {"aida": {"name": "  "}})
    assert naming.display_name(isolated_root) == "Aida"


@pytest.mark.asyncio
async def test_reconcile_writes_both_title_channels(isolated_root: Path) -> None:
    """Disk reconcile: the sidecar (picker) AND the transcript entry (resume).

    Both channels on the same event is what ``_persist_conversation_name``
    does for a live session; writing only one leaves the picker or a resume
    reading the old name — which is the class of bug the title sidecar was
    introduced to close.
    """
    directory = _her_session(isolated_root)
    write_config(isolated_root, {"aida": {"name": "Sovereign"}})

    applied = await naming.reconcile_session_title(isolated_root, HER_ID)
    assert applied == "Sovereign"
    assert stored_session_title(directory) == "Sovereign"
    title_state = read_title_state(directory)
    assert title_state is not None
    assert title_state.text == "Sovereign" and title_state.user_set is True
    journal = read_latest_custom(directory, "conversation_name")
    assert journal is not None and journal.get("text") == "Sovereign"


@pytest.mark.asyncio
async def test_reconcile_is_a_noop_when_the_title_already_matches(
    isolated_root: Path,
) -> None:
    directory = _her_session(isolated_root, title="Sovereign")
    write_config(isolated_root, {"aida": {"name": "Sovereign"}})

    assert await naming.reconcile_session_title(isolated_root, HER_ID) == "Sovereign"
    # Nothing was appended: the transcript's conversation_name channel stays
    # empty because the sidecar already agreed (an append here would be the
    # duplicate the live path documents it never makes).
    assert read_latest_custom(directory, "conversation_name") is None


@pytest.mark.asyncio
async def test_reconcile_refuses_an_invalid_explicit_name(isolated_root: Path) -> None:
    _her_session(isolated_root)
    assert await naming.reconcile_session_title(isolated_root, HER_ID, name="   ") is None
    assert stored_session_title(isolated_root / "sessions" / HER_ID) == "Aida"


@pytest.mark.asyncio
async def test_reconcile_skips_a_missing_session(isolated_root: Path) -> None:
    assert await naming.reconcile_session_title(isolated_root, "nosuchsession") is None


@pytest.mark.asyncio
async def test_creation_wears_the_configured_name(isolated_root: Path) -> None:
    """A first creation after a rename is BORN wearing the new name."""
    from local_operator import aida as aida_pkg

    write_config(isolated_root, {"aida": {"name": "Sovereign"}})
    her_id = await aida_pkg.ensure_session(isolated_root)
    assert her_id is not None
    directory = isolated_root / "sessions" / her_id
    assert stored_session_title(directory) == "Sovereign"
    journal = read_latest_custom(directory, "conversation_name")
    assert journal is not None and journal.get("text") == "Sovereign"


@pytest.mark.asyncio
async def test_ensure_reconciles_a_stale_title(isolated_root: Path) -> None:
    """The boot path rewrites an existing session's title from config.

    This is the not-live half: ``aida.name`` changed while no runtime had her
    session open (a /settings edit, a config write from another process), and
    the next ensure must point the STORED title at it — the picker and sidebar
    read disk, not the live session's memory.
    """
    from local_operator import aida as aida_pkg

    her_id = await aida_pkg.ensure_session(isolated_root)
    assert her_id is not None
    assert stored_session_title(isolated_root / "sessions" / her_id) == "Aida"

    write_config(isolated_root, {"aida": {"name": "Sovereign"}})
    assert await aida_pkg.ensure_session(isolated_root) == her_id
    assert stored_session_title(isolated_root / "sessions" / her_id) == "Sovereign"
    journal = read_latest_custom(isolated_root / "sessions" / her_id, "conversation_name")
    assert journal is not None and journal.get("text") == "Sovereign"
