"""Exercise team authoring through the real styled app and registry, not a bare host."""

from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry
from local_operator.tui.app import OperatorApp
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from tests.unit.tui.test_projects_view import _boot, _notices

pytestmark = pytest.mark.asyncio


class _PickerEditor:
    """The team's actual argument grammar, not the agent picker's command set."""

    def __init__(self, text):
        self.text = text
        self._argument_commands = ("team", "teams")
        self._command_names = frozenset({"team", "teams"})

    def _caret_offset(self):
        return len(self.text)


def setup(tmp_path, monkeypatch) -> tuple[Any, OperatorApp]:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    from local_operator.agents import AgentRegistry

    session = FakeSession()
    session.agent_registry = AgentRegistry(tmp_path)
    session.team_registry = TeamRegistry(tmp_path)
    return session, OperatorApp(lambda: _factory(session))


async def open_form(app, pilot, command="/team new"):
    app._run_slash_command(command)
    await pilot.pause()
    await pilot.pause()
    assert app._team_form is not None
    return app._team_form


async def test_real_app_create_edit_roundtrip(tmp_path, monkeypatch):
    session, app = setup(tmp_path, monkeypatch)
    async with app.run_test(size=(60, 24)) as pilot:
        await _boot(pilot, app)
        page = await open_form(app, pilot)
        page.controls["name"].value = "crew"
        page.controls["instructions"].text = "Coordinate."
        page.controls["members"].value = "coder:2, team:pod"
        page.controls["aliases"].value = "builders"
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert app._team_form is None
        row = session.team_registry.get_team_by_name("crew")
        assert row.instructions == "Coordinate."
        assert [(m.kind, m.role, m.count) for m in row.members] == [
            ("agent", "coder", 2),
            ("team", "pod", 1),
        ]
        page = await open_form(app, pilot, "/team edit builders")
        assert page.controls["name"].disabled
        assert page.controls["members"].value == "coder:2, team:pod"
        page.controls["label"].value = "Delivery"
        page.controls["instructions"].text = ""
        page.controls["project"].text = "Build."
        page.controls["aliases"].value = ""
        await pilot.press("ctrl+s")
        await pilot.pause()
        row = TeamRegistry(tmp_path).get_team(row.id)
        assert (row.label, row.instructions, row.project, row.aliases) == (
            "Delivery",
            "",
            "Build.",
            [],
        )
        assert any("saved team crew" in text for text in _notices(app))
        assert app._editor().can_focus
        assert app._transcript_view().display


async def test_validation_duplicate_noop_stale(tmp_path, monkeypatch):
    session, app = setup(tmp_path, monkeypatch)
    row = session.team_registry.create_team(TeamEditFields(name="crew", instructions="Help."))
    async with app.run_test(size=(60, 24)) as pilot:
        await _boot(pilot, app)
        page = await open_form(app, pilot)
        await pilot.press("ctrl+s")
        assert page.blocks["name"].error
        page.controls["name"].value = "crew"
        page.controls["instructions"].text = "Draft retained."
        await pilot.press("ctrl+s")
        assert page.blocks["name"].error
        assert "already exists" in str(page._receipt.render())
        assert app.focused is page.controls["name"]
        assert page.controls["instructions"].text == "Draft retained."
        app._close_team_form()
        await pilot.pause()
        page = await open_form(app, pilot, "/team edit crew")
        with monkeypatch.context() as context:
            context.setattr(
                session.team_registry,
                "update_team",
                lambda *args, **kwargs: pytest.fail("no-op wrote"),
            )
            await pilot.press("ctrl+s")
            assert "no changes" in str(page._receipt.render())
        # A different registry simulates an external writer under the real lock.
        TeamRegistry(tmp_path).update_team(row.id, TeamEditFields(description="Elsewhere"))
        page.controls["label"].value = "My edit"
        await pilot.press("ctrl+s")
        assert "Someone else changed this team" in str(page._receipt.render())
        assert page.controls["label"].value == "My edit"
        assert TeamRegistry(tmp_path).get_team(row.id).label != "My edit"


async def test_alias_duplicate_and_inline_field_validation(tmp_path, monkeypatch):
    session, app = setup(tmp_path, monkeypatch)
    session.team_registry.create_team(TeamEditFields(name="existing", aliases=["builders"]))
    async with app.run_test(size=(60, 24)) as pilot:
        await _boot(pilot, app)
        page = await open_form(app, pilot)
        page.controls["name"].value = "crew"
        page.controls["instructions"].text = "Help."
        for field, bad in (("manager", ""), ("members", "coder:no"), ("aliases", "a,,b")):
            before = page.controls[field].value
            page.controls[field].value = bad
            await pilot.press("ctrl+s")
            assert page.blocks[field].error
            assert app.focused is page.controls[field]
            page.controls[field].value = before
        page.controls["aliases"].value = "builders"
        await pilot.press("ctrl+s")
        assert page.blocks["aliases"].error
        assert app.focused is page.controls["aliases"]
        assert app._team_form is page


@pytest.mark.parametrize("interrupt", ["escape", "ctrl+c"])
async def test_dirty_discard_keyboard(tmp_path, monkeypatch, interrupt):
    _session, app = setup(tmp_path, monkeypatch)
    async with app.run_test(size=(60, 24)) as pilot:
        await _boot(pilot, app)
        page = await open_form(app, pilot)
        page.controls["project"].text = "Keep this draft"
        page.controls["project"].focus()
        for answer in ("n", "enter", "escape"):
            await pilot.press(interrupt)
            await pilot.pause()
            assert page._confirm
            assert 0 <= page._question.region.y < 24
            assert page._question.region.bottom <= 24
            focused = app.focused
            await pilot.press("tab", "shift+tab", "ctrl+c", "ctrl+c")
            assert page._confirm
            assert app.focused is focused
            assert app._last_interrupt_at == 0.0
            await pilot.press(answer)
            assert not page._confirm
            assert page.controls["project"].text == "Keep this draft"
        await pilot.press(interrupt, "y")
        await pilot.pause()
        assert app._team_form is None
        assert app._editor().can_focus


async def test_clean_ctrl_c_closes(tmp_path, monkeypatch):
    _session, app = setup(tmp_path, monkeypatch)
    async with app.run_test(size=(60, 24)) as pilot:
        await _boot(pilot, app)
        await open_form(app, pilot)
        await pilot.press("ctrl+c")
        assert app._team_form is None
        assert app._last_interrupt_at == 0.0


@pytest.mark.parametrize("mode", ["projects", "settings", "agent"])
async def test_sibling_replacement(tmp_path, monkeypatch, mode):
    _session, app = setup(tmp_path, monkeypatch)
    async with app.run_test(size=(60, 24)) as pilot:
        await _boot(pilot, app)
        page = await open_form(app, pilot)
        page.controls["instructions"].text = "Old draft"
        getattr(app, f"_open_{mode}_view" if mode != "agent" else "_open_agent_form")()
        await pilot.pause()
        assert app._team_form is None
        assert not page.is_attached
        getattr(app, f"_close_{mode}_view" if mode != "agent" else "_close_agent_form")()
        await pilot.pause()
        assert app._editor().can_focus
        assert app.focused is app._editor()
        assert app._transcript_view().display


async def test_reboot(tmp_path, monkeypatch):
    session, app = setup(tmp_path, monkeypatch)
    replacement = FakeSession()
    replacement.team_registry = session.team_registry
    async with app.run_test(size=(60, 24)) as pilot:
        await _boot(pilot, app)
        page = await open_form(app, pilot)
        page.controls["instructions"].text = "Old draft"
        monkeypatch.setattr(app, "_session_factory", lambda: _factory(replacement))
        await app._reload_session()
        await pilot.pause()
        assert app._team_form is None
        assert not page.is_attached
        assert app._session is replacement
        assert app._editor().can_focus
        assert app._transcript_view().display
        assert not app.screen.has_class("projects")


async def test_readonly_open_save_and_follower_does_not_route(tmp_path, monkeypatch):
    session, app = setup(tmp_path, monkeypatch)
    async with app.run_test(size=(60, 24)) as pilot:
        await _boot(pilot, app)
        page = await open_form(app, pilot)
        page.controls["name"].value = "crew"
        page.controls["instructions"].text = "Retain."
        session.owns_runtime = False
        await pilot.press("ctrl+s")
        assert "team changes" in str(page._receipt.render())
        assert session.team_registry.get_team_by_name("crew") is None
        app._close_team_form()
        session.owns_runtime = False
        session.frontend_state = SimpleNamespace(
            slash_capabilities=[SimpleNamespace(command="team", scope="authoritative_session")]
        )
        session.route_shared_slash = lambda *args, **kwargs: pytest.fail("authoring routed")
        for command in ("/team new", "/team edit crew"):
            app._run_slash_command(command)
            await pilot.pause()
            assert app._team_form is None
            assert "team changes" in _notices(app)[-1]


async def test_real_app_override_fixture(tmp_path, monkeypatch):
    session, app = setup(tmp_path, monkeypatch)
    row = session.team_registry.create_team(
        TeamEditFields(
            name="crew",
            instructions="Help.",
            members=[
                TeamMember(role="coder", tier="hi", effort="high"),
                TeamMember(role="reviewer", tier="lo", effort="auto"),
                TeamMember(kind="team", role="pod", tier="med", effort="low"),
            ],
        )
    )
    before = [m.model_dump(mode="json") for m in row.members]
    before_bytes = [m.model_dump_json().encode() for m in row.members]
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        page = await open_form(app, pilot, "/team edit crew")
        page.controls["description"].value = "Description only"
        await pilot.press("ctrl+s")
        after = TeamRegistry(tmp_path).get_team(row.id)
        assert [m.model_dump(mode="json") for m in after.members] == before
        assert [m.model_dump_json().encode() for m in after.members] == before_bytes
        page = await open_form(app, pilot, "/team edit crew")
        page.controls["members"].value = "coder:2, team:pod:2, designer"
        await pilot.press("ctrl+s")
        after = TeamRegistry(tmp_path).get_team(row.id)
        assert [m.model_dump(mode="json") for m in after.members] == [
            {**before[0], "count": 2},
            {**before[2], "count": 2},
            TeamMember(role="designer").model_dump(mode="json"),
        ]


async def test_reserved_routing_picker_and_printed_talk_command(tmp_path, monkeypatch):
    import re

    session, app = setup(tmp_path, monkeypatch)
    for name in ("new", "edit", "chart", "crew"):
        session.team_registry.create_team(TeamEditFields(name=name, instructions="Help."))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        rows = app._team_argument_choices(_PickerEditor("/team "))
        names = [row.name for row in rows]
        assert names[-3:] == ["chart", "new", "edit"]
        assert {"=new", "=edit", "=chart", "crew"} <= set(names[:-3])
        assert not app._team_argument_choices(_PickerEditor("/team new "))
        assert {row.name for row in app._team_argument_choices(_PickerEditor("/team edit "))} == {
            "edit new",
            "edit edit",
            "edit chart",
            "edit crew",
        }
        for name in ("new", "edit", "chart"):
            app._run_slash_command("/team =" + name)
            await pilot.pause()
            assert session.active_team_name == name
            assert app._team_form is None
            receipt = _notices(app)[-1]
            match = re.search(r"(/team \S+) <message>", receipt)
            assert match is not None
            command = match.group(1)
            assert command == "/team =" + name
            app._run_slash_command(command + " execute receipt")
            await pilot.pause()
            assert session.prompts[-1] == "execute receipt"
            assert session.active_team_name == name
        page = await open_form(app, pilot, "/team edit =chart")
        assert page.opening.name == "chart"
        app._close_team_form()
        await pilot.pause()
        app._run_slash_command("/team chart crew")
        await pilot.pause()
        assert app._org_chart_view is not None
        assert app._org_chart_view.team_name == "crew"
        assert all(verb in app._team_listing_footer() for verb in ("/team new", "/team edit"))


async def test_escape_strips_exactly_one_equals(tmp_path, monkeypatch):
    session, app = setup(tmp_path, monkeypatch)
    names = []

    def lookup(name):
        names.append(name)
        return None

    monkeypatch.setattr(session.team_registry, "get_team_by_name", lookup)
    async with app.run_test(size=(60, 24)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/team ==foo message")
        await pilot.pause()
        assert names == ["=foo"]
        assert "'=foo'" in _notices(app)[-1]


async def test_inline_manager_members_alias_and_create_instructions(tmp_path, monkeypatch):
    session, app = setup(tmp_path, monkeypatch)
    session.team_registry.create_team(TeamEditFields(name="other", aliases=["taken"]))
    async with app.run_test(size=(60, 24)) as pilot:
        await _boot(pilot, app)
        page = await open_form(app, pilot)
        page.controls["name"].value = "crew"
        page.controls["manager"].value = ""
        await pilot.press("ctrl+s")
        assert page.blocks["manager"].error
        assert app.focused is page.controls["manager"]
        page.controls["manager"].value = "director"
        await pilot.press("ctrl+s")
        assert page.blocks["instructions"].error
        assert app.focused is page.controls["instructions"]
        page.controls["instructions"].text = "Retain draft."
        page.controls["members"].value = "coder:no"
        await pilot.press("ctrl+s")
        assert page.blocks["members"].error
        assert app.focused is page.controls["members"]
        page.controls["members"].value = ""
        page.controls["aliases"].value = "one,,two"
        await pilot.press("ctrl+s")
        assert page.blocks["aliases"].error
        page.controls["aliases"].value = "taken"
        await pilot.press("ctrl+s")
        assert page.blocks["aliases"].error
        assert "already an alias" in str(page._receipt.render())
        assert app.focused is page.controls["aliases"]
        assert page.controls["instructions"].text == "Retain draft."


async def test_empty_registry_still_discovers_new(tmp_path, monkeypatch):
    _session, app = setup(tmp_path, monkeypatch)
    async with app.run_test(size=(60, 24)) as pilot:
        await _boot(pilot, app)
        assert [row.name for row in app._team_argument_choices(_PickerEditor("/team "))] == [
            "chart",
            "new",
            "edit",
        ]
        for command, expected in (
            ("/team edit", "/team edit <name>"),
            ("/team new extra", "/team new"),
        ):
            app._run_slash_command(command)
            await pilot.pause()
            assert app._team_form is None
            assert expected in _notices(app)[-1]


async def test_registry_disappearing_during_draft_refuses(tmp_path, monkeypatch):
    session, app = setup(tmp_path, monkeypatch)
    async with app.run_test(size=(60, 24)) as pilot:
        await _boot(pilot, app)
        page = await open_form(app, pilot)
        page.controls["name"].value = "crew"
        page.controls["instructions"].text = "Keep this draft."
        session.team_registry = None
        await pilot.press("ctrl+s")
        assert "No team registry" in str(page._receipt.render())
        assert page.controls["instructions"].text == "Keep this draft."
        assert app._team_form is page
