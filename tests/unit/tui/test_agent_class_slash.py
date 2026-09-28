"""``/agent class <name> proactive|reactive`` — the R36 switch in the TUI.

The storage half is pinned in ``tests/unit/test_action_class.py``; what these
pin is the slash surface: the reserved-word grammar (including the ``=``
escape), the report form, the flip's receipt, and the immediate cleanup call
on the session. Driven at the real ``OperatorApp`` dispatch, the way the other
slash tests are.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.action_class import class_from_tags
from local_operator.agents import AgentRegistry
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView
from tests.unit.tui.test_app_pilot import FakeSession, _factory


@pytest.fixture(autouse=True)
def _scratch_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    return tmp_path


def _notices(app: OperatorApp) -> list[str]:
    return [
        block._text
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
    ]


async def _settle(pilot, app: OperatorApp) -> list[str]:
    """Pause a bounded number of frames so the worker can land its notice.

    Fixed-count rather than stop-at-first-notice: boot itself may have painted
    a notice, and returning early would race the worker this test is about.
    """
    for _ in range(120):
        await pilot.pause()
    return _notices(app)


@pytest.mark.asyncio
async def test_the_report_form_names_the_class_without_changing_anything(
    tmp_path: Path,
) -> None:
    session = FakeSession()
    session.agent_registry = AgentRegistry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(40):
            await pilot.pause()
            if app._session is not None:
                break
        app._run_slash_command("/agent class aida")
        notices = await _settle(pilot, app)

    assert any("class proactive" in text for text in notices), notices
    # Seed resolution, not an install: the report must not write anything.
    assert AgentRegistry(tmp_path).get_agent_by_name("aida") is None


@pytest.mark.asyncio
async def test_the_flip_writes_the_tag_and_receipts_the_stop(tmp_path: Path) -> None:
    session = FakeSession()
    session.agent_registry = AgentRegistry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(40):
            await pilot.pause()
            if app._session is not None:
                break
        app._run_slash_command("/agent class aida reactive")
        notices = await _settle(pilot, app)
        for _ in range(100):  # the worker's registry write is a thread hop
            await pilot.pause()
            row = AgentRegistry(tmp_path).get_agent_by_name("aida")
            if row is not None:
                break

    assert any("now reactive" in text and "stopped" in text for text in notices), notices
    row = AgentRegistry(tmp_path).get_agent_by_name("aida")
    assert row is not None
    assert class_from_tags(row.tags) == "reactive"


@pytest.mark.asyncio
async def test_a_bad_class_word_is_refused_before_any_write(tmp_path: Path) -> None:
    session = FakeSession()
    session.agent_registry = AgentRegistry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(40):
            await pilot.pause()
            if app._session is not None:
                break
        app._run_slash_command("/agent class aida sideways")
        notices = await _settle(pilot, app)

    assert any("must be one of" in text for text in notices), notices
    assert AgentRegistry(tmp_path).get_agent_by_name("aida") is None
