"""Drive profile authoring through the shipped OperatorApp and a real registry."""

from pathlib import Path
from typing import Any

import pytest

from local_operator.agents import AgentRegistry
from local_operator.tools.agent_tool import AgentParams, write_profile
from local_operator.tui.app import OperatorApp
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from tests.unit.tui.test_projects_view import _boot, _notices

pytestmark = pytest.mark.asyncio


def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    session = FakeSession()
    session.agent_registry = AgentRegistry(tmp_path)
    return session, OperatorApp(lambda: _factory(session))


async def open_form(app: OperatorApp, pilot: Any, command="/agent new"):
    app._run_slash_command(command)
    await pilot.pause()
    await pilot.pause()
    assert app._agent_form is not None
    return app._agent_form


async def test_real_app_create_edit_roundtrip(tmp_path, monkeypatch):
    session, app = setup(tmp_path, monkeypatch)
    async with app.run_test(size=(60, 24)) as pilot:
        await _boot(pilot, app)
        page = await open_form(app, pilot)
        page.controls["name"].value = "writer"
        page.controls["instructions"].text = "Write clearly."
        page.controls["tools"].value = "read, bash"
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert app._agent_form is None
        row = session.agent_registry.get_agent_by_name("writer")
        assert row is not None
        assert session.agent_registry.get_agent_system_prompt(row.id) == "Write clearly."
        page = await open_form(app, pilot, "/agent edit writer")
        assert page.controls["tools"].value == "read, bash"
        assert page.controls["name"].disabled
        page.controls["label"].value = "Clear writer"
        page.controls["instructions"].text = "Write briefly."
        await pilot.press("ctrl+s")
        await pilot.pause()
        data = session.agent_registry.definition_revision_fields(row.id)
        assert data["label"] == "Clear writer"
        assert data["instructions"] == "Write briefly."
        assert data["tools"] == ["read", "bash"]
        assert any("saved agent writer" in sentence for sentence in _notices(app))


async def test_validation_duplicate_noop_and_revision(tmp_path, monkeypatch):
    session, app = setup(tmp_path, monkeypatch)
    write_profile(
        session.agent_registry,
        AgentParams(op="create", name="writer", instructions="Write."),
        creating=True,
    )
    async with app.run_test(size=(60, 24)) as pilot:
        await _boot(pilot, app)
        page = await open_form(app, pilot)
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert page.blocks["name"].error
        page.controls["name"].value = "writer"
        page.controls["instructions"].text = "Draft retained."
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert app._agent_form is page
        assert "already exists" in str(page._receipt.render())
        assert page.controls["instructions"].text == "Draft retained."
        app._close_agent_form()
        await pilot.pause()
        page = await open_form(app, pilot, "/agent edit writer")
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert "no changes" in str(page._receipt.render())
        write_profile(
            session.agent_registry,
            AgentParams(op="update", name="writer", label="Elsewhere"),
            creating=False,
        )
        page.controls["label"].value = "My edit"
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert "Someone else" in str(page._receipt.render())
        assert page.controls["label"].value == "My edit"


async def test_specialist_switch_and_discard_keyboard(tmp_path, monkeypatch):
    session, app = setup(tmp_path, monkeypatch)
    async with app.run_test(size=(60, 24)) as pilot:
        await _boot(pilot, app)
        page = await open_form(app, pilot)
        page.controls["tools"].value = "bash"
        page.controls["kind"].focus()
        await pilot.press("right")
        await pilot.pause()
        assert not page.blocks["tools"].display
        await pilot.press("left")
        await pilot.pause()
        assert page.blocks["tools"].display
        assert page.controls["tools"].value == "bash"
        page.controls["name"].value = "special"
        page.controls["instructions"].text = "Help."
        page.controls["kind"].focus()
        await pilot.press("right")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert page._confirm
        assert 0 <= page._question.region.y < 24
        focused = app.focused
        await pilot.press("tab", "shift+tab")
        assert app.focused is focused
        await pilot.press("enter")
        assert not page._confirm
        await pilot.press("ctrl+s")
        await pilot.pause()
        row = session.agent_registry.get_agent_by_name("special")
        data = session.agent_registry.definition_revision_fields(row.id)
        assert data["kind"] == "specialist"
        assert data["tools"] is None
        assert not data["delegate"]


async def test_readonly_and_packaged_seed_refusal(tmp_path, monkeypatch):
    session, app = setup(tmp_path, monkeypatch)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/agent edit coder")
        await pilot.pause()
        assert app._agent_form is None
        assert "Install this packaged starter" in _notices(app)[-1]
        monkeypatch.setattr(app, "_projects_read_only", lambda: True)
        app._run_slash_command("/agent new")
        await pilot.pause()
        assert app._agent_form is None
        assert "Read-only" in _notices(app)[-1]


async def test_reserved_and_escaped_names_and_autocomplete(tmp_path, monkeypatch):
    from tests.unit.tui.test_agent_class_slash import _PickerEditor

    session, app = setup(tmp_path, monkeypatch)
    for name in ("new", "edit", "=foo"):
        write_profile(
            session.agent_registry,
            AgentParams(op="create", name=name, instructions="Help."),
            creating=True,
        )
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        rows = app._agent_argument_choices(_PickerEditor("/agent "))
        names = [row.name for row in rows]
        assert names[-3:] == ["class", "new", "edit"]
        assert "=new" in names and "=edit" in names and "==foo" in names
        assert not app._agent_argument_choices(_PickerEditor("/agent new "))
        edit = app._agent_argument_choices(_PickerEditor("/agent edit "))
        assert any(row.name == "edit new" for row in edit)
        for escaped, actual in (("=new", "new"), ("=edit", "edit"), ("==foo", "=foo")):
            app._run_slash_command("/agent " + escaped)
            await pilot.pause()
            assert app._agent_form is None
            assert session.active_agent == actual


async def test_field_tab_order_and_readonly_write_refusal(tmp_path, monkeypatch):
    session, app = setup(tmp_path, monkeypatch)
    async with app.run_test(size=(60, 24)) as pilot:
        await _boot(pilot, app)
        page = await open_form(app, pilot)
        keys = list(page.controls)
        for key in keys:
            assert app.focused is page.controls[key]
            assert page.blocks[key].region.y >= page._fields.region.y
            await pilot.press("tab")
        assert app.focused is page.controls["name"]
        await pilot.press("shift+tab")
        assert app.focused is page.controls["action_class"]
        page.controls["name"].value = "should-not-write"
        page.controls["instructions"].text = "Keep the draft."
        session.owns_runtime = False
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert app._agent_form is page
        assert "Read-only" in str(page._receipt.render())
        assert session.agent_registry.get_agent_by_name("should-not-write") is None
        assert page.controls["instructions"].text == "Keep the draft."


async def test_clean_cancel_dirty_keep_then_discard(tmp_path, monkeypatch):
    _session, app = setup(tmp_path, monkeypatch)
    async with app.run_test(size=(60, 24)) as pilot:
        await _boot(pilot, app)
        editor = app._editor()
        page = await open_form(app, pilot)
        await pilot.press("escape")
        await pilot.pause()
        assert app._agent_form is None
        assert app.focused is editor
        page = await open_form(app, pilot)
        page.controls["name"].value = "unfinished"
        await pilot.press("escape")
        assert page._confirm
        await pilot.press("escape")
        assert not page._confirm
        assert page.controls["name"].value == "unfinished"
        await pilot.press("escape", "y")
        await pilot.pause()
        assert app._agent_form is None


async def test_noop_never_calls_writer_and_clear_refusals(tmp_path, monkeypatch):
    session, app = setup(tmp_path, monkeypatch)
    write_profile(
        session.agent_registry,
        AgentParams(op="create", name="writer", description="For drafts", instructions="Write."),
        creating=True,
    )
    async with app.run_test(size=(60, 24)) as pilot:
        await _boot(pilot, app)
        page = await open_form(app, pilot, "/agent edit writer")

        def forbidden(*args, **kwargs):
            raise AssertionError("no-op/refusal called persistence")

        monkeypatch.setattr("local_operator.tools.agent_tool.write_profile", forbidden)
        await pilot.press("ctrl+s")
        assert "no changes" in str(page._receipt.render())
        page.controls["description"].value = ""
        await pilot.press("ctrl+s")
        assert page.blocks["description"].error
        assert page.controls["description"].value == ""
        page.controls["description"].value = "For drafts"
        page.controls["instructions"].text = ""
        await pilot.press("ctrl+s")
        assert page.blocks["instructions"].error
        assert app._agent_form is page


async def test_follower_authoring_never_routes_to_owner(tmp_path, monkeypatch):
    from types import SimpleNamespace

    session, app = setup(tmp_path, monkeypatch)
    async with app.run_test(size=(60, 24)) as pilot:
        await _boot(pilot, app)
        session.owns_runtime = False
        session.frontend_state = SimpleNamespace(
            slash_capabilities=[SimpleNamespace(command="agent", scope="authoritative_session")]
        )

        def forbidden(*args, **kwargs):
            raise AssertionError("authoring routed to owner")

        session.route_shared_slash = forbidden
        for command in ("/agent new", "/agent edit coder"):
            app._run_slash_command(command)
            await pilot.pause()
            assert app._agent_form is None
            assert "Read-only" in _notices(app)[-1]
