"""A delayed runtime attachment updates data, never the user's interaction."""

from __future__ import annotations

import asyncio
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from textual import events

from local_operator.session.frontend_state import FrontendModelSpec
from local_operator.session.remote import RemoteSession
from local_operator.session.runtime import registry
from local_operator.session.runtime.server import RuntimeServer
from local_operator.tui.app import OperatorApp
from tests.unit.tui.test_cold_slash_binds import _never, _RoutingHandle
from tests.unit.tui.test_settings_view import _select


class _SelectionHandle(_RoutingHandle):
    """Record ordered commands and project their state over the real socket."""

    def __init__(self, session_id: str) -> None:
        super().__init__(session_id)
        self.prompt_states: list[dict[str, str]] = []
        self.mutation_reached = threading.Event()
        self.mutation_release = threading.Event()
        self.mutation_release.set()

    async def prompt(self, text, images=None, command_id=None):
        state = self._frontend.state
        self.prompt_states.append(
            {
                "model": state.selected_model.model_id if state.selected_model else "",
                "team": state.active_team,
                "agent": state.active_agent,
                "goal": state.goal,
            }
        )
        return await super().prompt(text, images, command_id)

    async def run_slash_authoritative(self, command, args, images):
        result = await super().run_slash_authoritative(command, args, images)
        self.mutation_reached.set()
        if not self.mutation_release.is_set():
            await asyncio.to_thread(self.mutation_release.wait)
        if command == "model":
            if args == "failing-selector":
                raise RuntimeError("synthetic mutation failure")
            provider, sep, model_id = args.partition("/")
            if not sep or not model_id:
                return {"kind": "notice", "text": "invalid model selector", "style": "warning"}
            spec = FrontendModelSpec(provider=provider, model_id=model_id)
            self._frontend.mutate(selected_model=spec, effective_model=spec)
        elif command == "team":
            self._frontend.mutate(active_team=args)
        elif command == "agent":
            self._frontend.mutate(active_agent=args)
        elif command == "goal":
            self._frontend.mutate(goal=args)
        elif command == "rename":
            self._frontend.mutate(conversation_title=args)
        return result


async def _until(pilot, predicate):
    async with asyncio.timeout(30):
        while not predicate():
            await pilot.pause()
            await asyncio.sleep(0.01)


@asynccontextmanager
async def _held_startup(tmp_path, monkeypatch, phase="sync"):
    for key in tuple(os.environ):
        if key.startswith("CMUX_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    config = tmp_path / "config"
    config.mkdir()
    (config / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: initial\n"
    )
    from local_operator.tui.settings import settings_reload

    settings_reload()
    handle = _SelectionHandle("startup-test")
    handle._frontend.mutate(
        model_catalogue=[
            {"provider": "test", "model_id": name, "connected": True} for name in ("model", "other")
        ]
    )
    server = RuntimeServer(handle, kind="tui")
    reached = asyncio.Event()
    release = asyncio.Event()

    async def engage(*args, **kwargs):
        if phase == "spawn":
            reached.set()
            await release.wait()
        server.start()
        marker = config / "sessions/startup-test/.session.pid"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(os.getpid()))
        async with asyncio.timeout(30):
            while not any(status == "live" for _, status in registry.scan(config)):
                await asyncio.sleep(0.01)

    viewer = await RemoteSession.cold(
        "startup-test", config_dir=config, cwd=str(tmp_path), takeover_factory=_never
    )
    original = viewer._await_frontend

    async def held_sync():
        sync = await original()
        if phase == "sync":
            reached.set()
            await release.wait()
        return sync

    async def factory():
        return viewer

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", engage)
    monkeypatch.setattr(viewer, "_await_frontend", held_sync)
    monkeypatch.setattr(OperatorApp, "_check_for_update", lambda self: None)
    app = OperatorApp(factory, warm_session_imports=False)
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            await _until(pilot, reached.is_set)
            try:
                yield app, pilot, viewer, handle, release
            finally:
                # Gates must open BEFORE run_test tears down: a retire request
                # otherwise queues behind the held mutation on the real socket.
                handle.mutation_release.set()
                release.set()
    finally:
        release.set()
        await viewer.dispose()
        server.close()
        # The next test may use another HOME without this fixture. Leave no
        # committed display preference cached after the isolated app is gone.
        settings_reload()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["draft", "model", "team", "agent", "settings"])
async def test_sync_preserves_active_interaction(surface, tmp_path, monkeypatch):
    async with _held_startup(tmp_path, monkeypatch) as (app, pilot, viewer, handle, release):
        editor = app._editor()
        if surface == "draft":
            await pilot.press(*"hello world", "left", "left", "shift+left", "shift+left")
        elif surface == "settings":
            app._run_slash_command("/settings")
            await pilot.pause()
            await pilot.press("enter", *"open")
        else:
            await pilot.press(*f"/{surface} ", "down")
        await pilot.pause()
        before = (editor.text, editor.selection, app.focused)
        picker = editor.model_picker
        settings = app._settings_view
        settings_state = (
            (settings._selected, settings._editing, settings._buffer) if settings else None
        )
        assert viewer.is_cold
        release.set()
        await _until(pilot, lambda: not viewer.is_cold and not app._starting_runtime)
        assert app._editor() is editor
        assert (editor.text, editor.selection, app.focused) == before
        assert app._settings_view is settings
        if settings is not None:
            assert (settings._selected, settings._editing, settings._buffer) == settings_state
        if surface == "model":
            assert editor.model_picker is picker
            assert {row.selector for row in picker.rows()} == {"test/model", "test/other"}
            assert app._welcome is not None
            assert app._welcome._info.model_label == viewer.effective_model_label
            await pilot.press("down")
            held = picker.highlighted_selector()
            # A subsequent catalogue expansion retains an intentional highlight.
            handle._frontend.mutate(
                model_catalogue=handle._frontend.state.model_catalogue
                + [{"provider": "test", "model_id": "new", "connected": True}]
            )
            await _until(pilot, lambda: len(picker.rows()) == 3)
            assert picker.highlighted_selector() == held
            await pilot.press("escape")
            handle._frontend.mutate(model_catalogue=[])
            await _until(pilot, lambda: viewer.owner_model_catalogue() == [])
            assert not picker.is_open(), "late data must not reopen an Esc-dismissed picker"
        await pilot.press("escape", "escape", "Z")
        assert "Z" in editor.text


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["model", "team", "agent", "goal", "rename"])
@pytest.mark.parametrize("phase", ["spawn", "sync"])
async def test_latest_committed_selection_survives_sync(command, phase, tmp_path, monkeypatch):
    async with _held_startup(tmp_path, monkeypatch, phase) as (app, pilot, viewer, handle, release):
        for choice in ("first", "latest"):
            target = f"test/{choice}" if command == "model" else choice
            app.post_message(events.Paste(f"/{command} {target}"))
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
        await pilot.press(*"next draft", "left", "left")
        editor = app._editor()
        before = (editor.text, editor.selection, app.focused)
        assert handle.calls == [], "the old sync must precede every committed selection"
        release.set()
        await _until(
            pilot,
            lambda: len([call for call in handle.calls if call[0] == "run_slash_authoritative"])
            == 2,
        )
        await _until(
            pilot,
            lambda: (
                viewer.model.model_id == "latest"
                if command == "model"
                else getattr(
                    viewer.frontend_state,
                    {
                        "team": "active_team",
                        "agent": "active_agent",
                        "goal": "goal",
                        "rename": "conversation_title",
                    }[command],
                )
                == "latest"
            ),
        )
        calls = [call[1][:2] for call in handle.calls if call[0] == "run_slash_authoritative"]
        assert calls == [
            (command, "test/first" if command == "model" else "first"),
            (command, "test/latest" if command == "model" else "latest"),
        ]
        assert (editor.text, editor.selection, app.focused) == before


@pytest.mark.asyncio
async def test_committed_setting_and_new_edit_survive_sync(tmp_path: Path, monkeypatch):
    from local_operator.tui.settings import settings_get

    async with _held_startup(tmp_path, monkeypatch) as (app, pilot, viewer, handle, release):
        app._run_slash_command("/settings")
        await pilot.pause()
        settings = app._settings_view
        assert settings is not None
        _select(settings, "display.comfortable_rows")
        previous = settings_get("display.comfortable_rows")
        await pilot.press("space")
        await pilot.pause()
        saved = settings_get("display.comfortable_rows")
        assert saved is not previous, "Space must commit the changed setting"
        from local_operator.config import ConfigManager

        assert (
            ConfigManager(tmp_path / "config").get_config_value("display.comfortable_rows") == saved
        )
        _select(settings, "hosting")
        await pilot.press("enter", *"new")
        before = (settings._selected, settings._editing, settings._buffer, app.focused)
        release.set()
        await _until(pilot, lambda: not viewer.is_cold and not app._starting_runtime)
        assert settings_get("display.comfortable_rows") == saved
        assert (settings._selected, settings._editing, settings._buffer, app.focused) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["spawn", "sync"])
async def test_picker_choice_and_saved_use_canonical_dispatch(phase, tmp_path, monkeypatch):
    from local_operator.config import ConfigManager
    from local_operator.tui.widgets.model_picker import ModelRow

    async with _held_startup(tmp_path, monkeypatch, phase) as (app, pilot, viewer, handle, release):
        editor = app._editor()
        # Local cached rows can be available before the owner's first catalogue.
        app._run_slash_command("/model")
        await pilot.pause()
        editor.model_picker.set_rows(
            [
                ModelRow(provider="test", model_id=name, label=name, connected=True)
                for name in ("initial", "picked")
            ],
            current="test/initial",
        )
        await pilot.press("down", "enter")
        await pilot.pause()
        assert handle.calls == [], "a real picker choice must not call a raw remote setter"
        # Saved is resolved on THIS terminal at commitment, not on the owner or
        # at some later bind callback after the config has changed again.
        config = ConfigManager(tmp_path / "config")
        config.set_config_value("model_name", "saved-choice")
        app._run_slash_command("/model saved")
        config.set_config_value("model_name", "later-default")
        await pilot.press(*"next draft", "left")
        before = (editor.text, editor.selection, app.focused)
        release.set()
        await _until(pilot, lambda: viewer.model.model_id == "saved-choice")
        calls = [call[1][:2] for call in handle.calls if call[0] == "run_slash_authoritative"]
        assert calls == [("model", "test/picked"), ("model", "test/saved-choice")]
        assert (editor.text, editor.selection, app.focused) == before


@pytest.mark.asyncio
async def test_cold_mutation_census_uses_canonical_locality(tmp_path, monkeypatch):
    async with _held_startup(tmp_path, monkeypatch) as (app, pilot, viewer, handle, release):
        # This table pins argument-dependent LOCAL exceptions, not a second
        # authority registry. All other commands inherit canonical locality.
        for command, arg in [
            ("goal", "a goal"),
            ("rename", "a title"),
            ("effort", "high"),
            ("fast", ""),
            ("approvals", "ask"),
            ("loop", ""),
            ("compact", ""),
            ("mcp", "reauth demo"),
            ("model", "test/next"),
            ("team", "chosen"),
            ("agent", "chosen"),
            ("credential", "list"),
        ]:
            assert app._needs_runtime_first(f"/{command}", arg), (command, arg)
        for command, arg in [
            ("model", ""),
            ("model", "default test/next"),
            ("team", ""),
            ("team", "chart example"),
            ("agent", ""),
            ("goal", ""),
            ("context", ""),
            ("effort", ""),
            ("approvals", ""),
            ("mcp", ""),
            ("settings", ""),
            ("stop", ""),
            ("theme", "dark"),
        ]:
            assert not app._needs_runtime_first(f"/{command}", arg), (command, arg)


async def _enter_line(app, pilot, text):
    app.post_message(events.Paste(text))
    await pilot.pause()
    await pilot.press("escape", "enter")
    await pilot.pause()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["spawn", "sync", "bound"])
@pytest.mark.parametrize("command", ["model", "team", "agent", "goal"])
async def test_committed_mutations_apply_before_following_prompt(
    phase, command, tmp_path, monkeypatch
):
    async with _held_startup(tmp_path, monkeypatch, "sync" if phase == "bound" else phase) as (
        app,
        pilot,
        viewer,
        handle,
        release,
    ):
        if phase == "bound":
            release.set()
            await _until(pilot, lambda: not viewer.is_cold and not app._starting_runtime)
        # Hold actual mutation application/ack, not merely its dispatch. Prompt
        # state is sampled in the owner's prompt method on the real socket.
        handle.mutation_release.clear()
        for value in ("first", "latest"):
            arg = f"test/{value}" if command == "model" else value
            await _enter_line(app, pilot, f"/{command} {arg}")
        await _enter_line(app, pilot, "following prompt")
        await pilot.press(*"second draft")
        release.set()
        await _until(pilot, handle.mutation_reached.is_set)
        assert handle.prompt_states == [], "a prompt cannot ingest the pre-mutation state"
        handle.mutation_release.set()
        await _until(pilot, lambda: len(handle.prompt_states) == 1)
        assert handle.prompt_states[0][command] == "latest"
        assert [call[0] for call in handle.calls] == [
            "run_slash_authoritative",
            "run_slash_authoritative",
            "prompt",
        ]
        assert app._editor().text == "second draft"


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["spawn", "sync"])
@pytest.mark.parametrize("source", ["picker", "saved"])
async def test_model_choice_source_applies_before_following_prompt(
    phase, source, tmp_path, monkeypatch
):
    from local_operator.config import ConfigManager
    from local_operator.tui.widgets.model_picker import ModelRow

    async with _held_startup(tmp_path, monkeypatch, phase) as (app, pilot, viewer, handle, release):
        handle.mutation_release.clear()
        if source == "picker":
            app._run_slash_command("/model")
            await pilot.pause()
            app._editor().model_picker.set_rows(
                [
                    ModelRow(provider="test", model_id=name, label=name, connected=True)
                    for name in ("initial", "latest")
                ],
                current="test/initial",
            )
            await pilot.press("down", "enter")
            await pilot.pause()
        else:
            ConfigManager(tmp_path / "config").set_config_value("model_name", "latest")
            await _enter_line(app, pilot, "/model saved")
        await _enter_line(app, pilot, "following prompt")
        release.set()
        await _until(pilot, handle.mutation_reached.is_set)
        assert handle.prompt_states == []
        handle.mutation_release.set()
        await _until(pilot, lambda: len(handle.prompt_states) == 1)
        assert handle.prompt_states[0]["model"] == "latest"


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["spawn", "sync", "bound"])
@pytest.mark.parametrize("selector", ["invalid-selector", "failing-selector"])
async def test_refused_choice_keeps_previous_state_and_releases_prompt(
    phase, selector, tmp_path, monkeypatch
):
    from tests.unit.tui.test_app_pilot import _transcript_text

    async with _held_startup(tmp_path, monkeypatch, "sync" if phase == "bound" else phase) as (
        app,
        pilot,
        viewer,
        handle,
        release,
    ):
        if phase == "bound":
            release.set()
            await _until(pilot, lambda: not viewer.is_cold and not app._starting_runtime)
        handle.mutation_release.clear()
        await _enter_line(app, pilot, f"/model {selector}")
        await _enter_line(app, pilot, "following refusal")
        release.set()
        await _until(pilot, handle.mutation_reached.is_set)
        assert handle.prompt_states == []
        handle.mutation_release.set()
        await _until(pilot, lambda: len(handle.prompt_states) == 1)
        assert handle.prompt_states[0]["model"] == "model"
        message = (
            "invalid model selector"
            if selector == "invalid-selector"
            else "synthetic mutation failure"
        )
        await _until(pilot, lambda: message in _transcript_text(app))


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["spawn", "sync"])
@pytest.mark.parametrize("navigation", ["new", "resume"])
async def test_session_navigation_retires_queued_startup_choice(
    phase, navigation, tmp_path, monkeypatch
):
    from tests.unit.tui.test_app_pilot import FakeSession, _transcript_text

    async with _held_startup(tmp_path, monkeypatch, phase) as (app, pilot, viewer, handle, release):
        replacement = FakeSession()

        async def resume_factory(session_id):
            return replacement

        app._resume_factory = resume_factory
        await _enter_line(app, pilot, "/model test/stale")
        app._run_slash_command("/new" if navigation == "new" else "/resume replacement")
        await _until(pilot, lambda: app._session is replacement)
        release.set()
        await pilot.pause()
        assert handle.calls == [], "a cancelled choice cannot reach the outgoing owner"
        assert "owner ran /model" not in _transcript_text(app)


@pytest.mark.asyncio
async def test_inflight_mutation_receipt_does_not_paint_a_replacement(tmp_path, monkeypatch):
    from tests.unit.tui.test_app_pilot import FakeSession, _transcript_text

    async with _held_startup(tmp_path, monkeypatch) as (app, pilot, viewer, handle, release):
        release.set()
        await _until(pilot, lambda: not viewer.is_cold and not app._starting_runtime)
        replacement = FakeSession()

        async def resume_factory(session_id):
            return replacement

        app._resume_factory = resume_factory
        handle.mutation_release.clear()
        await _enter_line(app, pilot, "/model test/old-session-only")
        await _until(pilot, handle.mutation_reached.is_set)
        app._run_slash_command("/new")
        await pilot.pause()
        handle.mutation_release.set()
        await _until(pilot, lambda: app._session is replacement)
        await pilot.pause()
        assert "owner ran /model" not in _transcript_text(app)
        assert "old-session-only" not in replacement.model_label
