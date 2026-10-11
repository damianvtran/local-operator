"""``/language``: the command surface (slice C).

The command is the discoverable half of the ``language`` setting; its tests
are about the four shapes a user can meet — bare (what am I on), a write
(persists, receipts), a no-op (already there), and a refusal (unknown, or a
supported locale no wave has shipped yet). The LIVE half — a switch or a
foreign write moving what the running TUI resolves — is
``test_language_live.py``; both files share the isolated config dir, so a
committed switch never touches the operator's real config.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.config import ConfigManager
from local_operator.paths import CONFIG_DIR_ENV
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.transcript import NoticeBlock
from tests.unit.tui.test_app_pilot import FakeSession, _factory


@pytest.fixture()
def config_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the config at a temp dir — a committed switch WRITES one."""
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    return tmp_path


async def _boot(pilot, app: OperatorApp) -> None:
    for _ in range(40):
        await pilot.pause()
        if app._session is not None:
            return


def _notices(app: OperatorApp) -> list[str]:
    return [block.text() or "" for block in app.query(NoticeBlock)]


def _stored(config_dir: Path) -> str | None:
    value = ConfigManager(config_dir).get_config().values.get("language")
    return value if isinstance(value, str) else None


@pytest.mark.asyncio
async def test_bare_language_names_the_active_language(config_dir: Path) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/language")
        await pilot.pause()
        texts = _notices(app)
        # Resolved language, its English name, the value space, and the
        # pointer at the /settings row — the operator's "the settings path
        # plus a discoverable command affordance".
        assert any(
            text.startswith("language: en (English)")
            and "<auto|en>" in text
            and "/settings" in text
            for text in texts
        ), texts


@pytest.mark.asyncio
async def test_a_switch_persists_the_same_key_the_settings_row_writes(
    config_dir: Path,
) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/language en")
        await pilot.pause()
        # ONE truth: the registry key the /settings row owns, written through
        # the same facade (`settings_io.write_setting`).
        assert _stored(config_dir) == "en"
        assert any(
            text.startswith("language: en saved — active en (English)") for text in _notices(app)
        ), _notices(app)


@pytest.mark.asyncio
async def test_switching_to_auto_when_already_auto_writes_nothing(
    config_dir: Path,
) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/language auto")
        await pilot.pause()
        # No config value existed; `auto` IS the absent value, so the no-op
        # receipt must not flip it into an explicitly stored one.
        assert _stored(config_dir) is None
        assert any("language: already auto" in text for text in _notices(app)), _notices(app)


@pytest.mark.asyncio
async def test_unknown_and_unshipped_values_are_refused_with_the_option_list(
    config_dir: Path,
) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/language klingon")
        await pilot.pause()
        assert any(
            "unknown language 'klingon' — available: auto, en" in text for text in _notices(app)
        ), _notices(app)
        # `fr` is SUPPORTED but not SHIPPED: an unaudited locale keys no
        # catalogue, so it must be refused the same way, and the option list
        # is the honest one (auto + the shipped set).
        app._run_slash_command("/language fr")
        await pilot.pause()
        assert any(
            "unknown language 'fr' — available: auto, en" in text for text in _notices(app)
        ), _notices(app)
        assert _stored(config_dir) is None
