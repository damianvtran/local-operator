"""The actual /memory slash flow with a bound session and registry-team identity.

A context-free FakeSession hid a missing Path import: the real context path crashed
before showing any non-global fact. Keep a bound fake, never a live operator session.
"""

from pathlib import Path

import pytest

from local_operator.harness.types import ToolContext
from local_operator.memory import MemoryIdentity, MemoryStore
from local_operator.teams import TeamEditFields, TeamRegistry
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.transcript import NoticeBlock
from tests.unit.tui.test_app_pilot import FakeSession, _factory


@pytest.mark.asyncio
async def test_bound_session_and_team_list_show_and_candidate_correction(tmp_path):
    root = tmp_path / "config"
    team = TeamRegistry(root).create_team(TeamEditFields(name="synthetic-memory-team"))

    class BoundSession(FakeSession):
        def _build_tool_context(self):
            return ToolContext(
                session_id=self.session_id,
                team_id=team.id,
                agent_id="main",
                memory_config_dir=str(root),
                session_dir=str(root / "sessions" / self.session_id),
            )

    session = BoundSession()
    context = session._build_tool_context()
    assert context.session_dir is not None
    store = MemoryStore(
        root,
        MemoryIdentity(context.session_id, context.team_id, context.agent_id),
        Path(context.session_dir),
        origin="synthetic-tui",
    )
    item = store.put(
        "session", {"title": "Captured runbook", "body": "docs/r.md", "status": "candidate"}
    )
    store.put("team", {"title": "Team decision", "body": "Ship Friday"})
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 38)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()

        async def submit(text):
            editor = app.query_one(Editor)
            editor.focus()
            editor.load_text(text)
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()

        await submit("/memory list")
        assert any("global, session, team" in block.text() for block in app.query(NoticeBlock))
        await submit("/memory show session/" + item["id"])
        assert any("Captured runbook" in block.text() for block in app.query(NoticeBlock))
        await submit(
            f"/memory edit session/{item['id']} --revision {item['version']} "
            "--body 'Corrected runbook'"
        )
        current = store.show("session", item["id"])
        assert current["fact"]["body"] == "Corrected runbook"
        assert current["status"] == "active"
